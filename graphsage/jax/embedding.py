from jax.nn.initializers import uniform
import jax


def Embedding(
    vocab_size,
    embedding_size,
    padding_idx=None,
    embedding_init=uniform(),
    fill_with=None,
):
    """Layer construction function for an embedding layer."""

    def init_fun(rng, input_shape):
        embedding_shape = (vocab_size, embedding_size)

        rng, k1 = jax.random.split(rng)
        embedding_table = embedding_init(k1, embedding_shape)
        if padding_idx is not None:
            embedding_table = index_update(embedding_table, padding_idx, 0.0)
        output_shape = input_shape[:-1] + (embedding_size,)

        if fill_with is not None:
            embedding_table = fill_with

        return output_shape, (embedding_table,)

    def apply_fun(params, inputs, **kwargs):

        embedding_table = params[0]
        return embedding_table[inputs, :]

    return init_fun, apply_fun
