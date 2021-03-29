import jax.numpy as jnp
from jax import random
from jax.nn.initializers import xavier_uniform
import jax.nn as nn
import operator
import random as rando

import jax

from embedding import Embedding


def node_encoder(embed_dim, gcn=False):
    """
    Args:
        features: The features for each node in the dataset
        embed_dim: The dimension for the final node embeddings
        feat_dim: the dimensionality of the features
        gcn: whether to perform graph convolution
    """

    def init_fn(rng, input_shape):

        output_shape = input_shape[:-1] + (embed_dim,)

        W_init = xavier_uniform()
        k1, k2 = random.split(rng)

        # Weight matrix feature dimension -> embedding_dimension.
        # The input dimension is equal to the input shape of the layer if
        # gcn=True, otherwise if needs to be doubled as the input and neighbor
        # representations are concatenated for graphsage.
        W = W_init(k2, (embed_dim, input_shape[1] if gcn else 2 * input_shape[1]))

        return output_shape, (W,)

    def apply_fn(
        params, inputs, all_feats, idx, adj, aggregator_fn, num_sample=10, rng=None
    ):

        rng, k2 = random.split(rng)
        # Aggregate features from neighbouring nodes.
        neigh_feats = aggregator_fn(
            all_feats,
            idx,
            [set(adj[node.astype(int)]) for node in idx],
            gcn,
            rng=k2,
            num_sample=num_sample,
        )

        if not gcn:
            # If not in GCN mode, select features for the current nodes
            self_feats = inputs
            # Concatenate features with aggregational stuff
            combined = jnp.concatenate([self_feats, neigh_feats], axis=1)
        else:

            # If in GCN mode, use neighbour features
            combined = neigh_feats

        # Multiply resulting features by weight matrix.
        combined = nn.relu(combined.dot(params[0].T))
        return combined

    return init_fn, apply_fn


def graphsage(num_classes, features, embed_dim, gcn=False, rng=None):

    # The model receives the embedding layer for the input features
    init_emb, apply_emb = Embedding(
        features.shape[0], features.shape[1], fill_with=features
    )

    # The model receives the encoder
    init_enc, apply_enc = node_encoder(embed_dim=embed_dim, gcn=gcn)

    # Call the input embedding init funcion. This creates an embedding layer
    # containing the input features
    emb_shape, emb_params = init_emb(
        rng,
        features.shape,
    )

    def init_fn(rng, input_shape):

        # Call the encoder init functions.
        # This returns the output shape of the encoder and its parameters.
        enc_shape, enc_params = init_enc(rng, emb_shape)

        # (n_nodes, n_encoder_features) -> (n_nodes, num_classes)
        output_shape = enc_shape[:-1] + (num_classes,)

        rng, k2 = random.split(rng)
        W_init = xavier_uniform()

        # Weight matrix to project output of encoder into into output scores
        W = W_init(k2, (enc_shape[-1], num_classes))

        # Returns both the embedding, encoder parameters and the current parameters
        return output_shape, (*enc_params, W)

    def apply_fn(params, idx, adj, aggregator_fn, num_sample=10):
        """
        Args:
            params: tuple of model parameters
            idx: list of node indices
            adj: the adjacency lists
            aggregator_fn: Aggregator funcion
            num_sample: Numbers of samples in a batch. Defaults to 10.

        Returns:
            the output scores (n_nodes, n_classes)
        """

        enc_params = params[0]

        input_embeds = apply_emb(emb_params, idx)

        # Get embeddings from encoder
        embeds = apply_enc(
            (enc_params,),
            input_embeds,
            features,
            idx,
            adj,
            aggregator_fn,
            num_sample,
            rng=rng,
        )

        # Multiply embeddings by weight matrix and return result
        scores = embeds.dot(params[-1])
        return scores

    return init_fn, apply_fn


def aggregator(features, nodes, to_neighs, gcn, rng, num_sample=10):
    """
    to_neighs: one list for each node in nodes, contains indices of the neighbors
    """

    _set = set
    if not num_sample is None:

        idxs = jax.random.randint(rng, (num_sample,), 0, len(to_neighs))

        samp_neighs = [
            _set(jnp.array(list(to_neigh))[idxs])
            if len(to_neigh) >= num_sample
            else to_neigh
            for to_neigh in to_neighs
        ]
    else:
        samp_neighs = to_neighs

    if gcn:
        for i, samp_neigh in enumerate(samp_neighs):
            samp_neigh.add(int(nodes[i]))

    unique_nodes_list = list(set.union(*samp_neighs))

    unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
    mask = jnp.zeros((len(samp_neighs), len(unique_nodes)))
    column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
    row_indices = [
        i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))
    ]

    mask = jax.ops.index_update(mask, jax.ops.index[row_indices, column_indices], 1)

    num_neigh = mask.sum(1, keepdims=True)
    mask = mask / num_neigh
    embed_matrix = features[unique_nodes_list, :]
    to_feats = mask.dot(embed_matrix)
    return to_feats
