import argparse
import time

import numpy
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers

from model import graphsage, aggregator, node_encoder
from embedding import Embedding

import os, sys, inspect

from tqdm import tqdm, trange

import sys
sys.path.append("../../")

from common.data import load_data

def loss(params, batch, aggregator_fn):
    """
    The idxes of the batch indicate which nodes are used to compute the loss.
    """

    def softmax(X):
        exps = jnp.exp(X - jnp.max(X, axis=1).reshape(-1, 1))
        return exps / jnp.sum(exps, axis=1)[:, None]

    def cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray
        Returns: scalar
        """
        predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
        N = predictions.shape[0]
        ce = -jnp.sum(targets * jnp.log(predictions + 1e-9)) / N
        return ce

    inputs, targets, adj, rng, idx = batch
    preds = predict_fun(params, idx, adj, aggregator_fn)
    preds = softmax(preds)

    ce_loss = cross_entropy(preds, targets[idx])
    return ce_loss


def accuracy(params, batch, aggregator_fn):
    inputs, targets, adj, rng, idx = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict_fun(params, idx, adj, aggregator_fn), axis=1)
    return jnp.mean(predicted_class == target_class[idx])


def loss_accuracy(params, batch, aggregator_fn):
    def softmax(X):
        exps = jnp.exp(X - jnp.max(X, axis=1).reshape(-1, 1))
        return exps / jnp.sum(exps, axis=1)[:, None]

    def cross_entropy(predictions, targets, epsilon=1e-12):
        """
        Computes cross entropy between targets (encoded as one-hot vectors)
        and predictions.
        Input: predictions (N, k) ndarray
            targets (N, k) ndarray
        Returns: scalar
        """
        predictions = jnp.clip(predictions, epsilon, 1.0 - epsilon)
        N = predictions.shape[0]
        ce = -jnp.sum(targets * jnp.log(predictions + 1e-9)) / N
        return ce

    inputs, targets, adj, rng, idx = batch
    preds = predict_fun(params, idx, adj, aggregator_fn)
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(preds, axis=1)
    ce_loss = cross_entropy(softmax(preds), targets[idx])
    acc = jnp.mean(predicted_class == target_class[idx])
    return ce_loss, acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--embed_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    args = parser.parse_args()

    rng_key = random.PRNGKey(args.seed)
    rng_key, k2 = random.split(rng_key)

    ### Dataset handling
    adj, features, labels, idx_train, idx_val, idx_test = load_data("../../",
                                                                    mode="sage")

    idx_train = jnp.array(idx_train)
    idx_val = jnp.array(idx_val)
    labels = jnp.array(labels)
    features = jnp.array(features)

    num_nodes = 2708
    rand_indices = jax.random.permutation(k2, num_nodes)
    idx_test = rand_indices[:1000]
    idx_val = rand_indices[1000:1500]
    idx_train = list(rand_indices[1500:])

    ### Argument parsing
    step_size = args.lr
    num_epochs = args.epochs
    n_nodes = features.shape[0]
    n_feats = features.shape[1]

    init_fun, predict_fun = graphsage(
        num_classes=7,
        features=features,
        embed_dim=args.embed_size,
        gcn=False,
        rng=rng_key,
    )

    input_shape = (-1, n_nodes, features.shape[1])
    rng_key, init_key = random.split(rng_key)
    _, init_params = init_fun(init_key, input_shape)

    opt_init, opt_update, get_params = optimizers.adam(step_size)

    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch, aggregator), opt_state)

    opt_state = opt_init(init_params)
    print("\nStarting training...")

    num_batch = 100
    t = trange(num_batch, leave=True)
    for batch_i in t:

        batch_idx = idx_train[:256]

        # Reshuffle training idx
        rng_key, k2 = random.split(rng_key)
        idx_train = random.shuffle(k2, jnp.array(idx_train))

        # Get batch and update params
        batch = (features, labels, adj, rng_key, jnp.array(batch_idx))
        opt_state = update(batch_i, opt_state, batch)

        params = get_params(opt_state)
        train_loss, train_acc = loss_accuracy(params, batch, aggregator)

        eval_batch = (features, labels, adj, rng_key, jnp.array(idx_val))
        val_loss, val_acc = loss_accuracy(params, eval_batch, aggregator)

        if batch_i % 5 == 0:
            t.set_description(
                f"Iter {batch_i}/{num_batch}  train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} val_loss: {val_loss} val_acc: {val_acc}"
            )

        # Generate new random key cause why not
        rng_key, _ = random.split(rng_key)

    # now run on the test set
    test_batch = (features, labels, adj, rng_key, jnp.array(idx_test))
    test_acc = accuracy(params, test_batch, aggregator)
    print(f"Test set acc: {test_acc}")
