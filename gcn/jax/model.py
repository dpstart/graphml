import math

import jax
import jax.numpy as jnp
from jax import lax, random
import jax.nn as nn
from jax.nn.initializers import glorot_uniform, zeros
import os, sys


# TODO: allow sparse matrix multiply
def GraphConv(out_dim, bias=False, sparse=False):
    def matmul(a, b):
        return jnp.matmul(a, b)

    def init_fn(rng, input_shape):

        output_shape = input_shape[:-1] + (out_dim,)
        rng, k = random.split(rng)
        W_init, b_init = glorot_uniform(), zeros
        W = W_init(k, (input_shape[-1], out_dim))

        b = b_init(k, (out_dim,)) if bias else None

        return output_shape, (W, b)

    def apply_fn(params, x, adj, **kwargs):

        W, b = params
        support = x @ W
        out = matmul(adj, support)

        if bias:
            out += b
        return out

    return init_fn, apply_fn


# TODO: sparsify
def GCN(nhid, nclasses, sparse=False):
    """
    Two-layers Graph Convolutional Network
    """

    init_fn1, apply_fn1 = GraphConv(nhid, bias=False)
    init_fn2, apply_fn2 = GraphConv(nclasses, bias=False)

    init_fn_list = [init_fn1, init_fn2]
    apply_fn_list = [apply_fn1, apply_fn2]

    def init_fn(rng, input_shape):

        params = []
        # output_shape = input_shape[:-1] + (nclasses,)
        for init_fn_ in init_fn_list:
            rng, k = random.split(rng)
            input_shape, params_ = init_fn_(k, input_shape)
            params.append(params_)
        output_shape = input_shape
        return output_shape, params

    def apply_fn(params, x, adj, **kwargs):

        for (i,apply_fn_) in enumerate(apply_fn_list[:-1]):

            x = apply_fn_(params[i], x, adj)
            x = nn.relu(x)

        x = apply_fn_list[-1](params[-1], x, adj)
        x = nn.log_softmax(x)
        return x

    return init_fn, apply_fn
