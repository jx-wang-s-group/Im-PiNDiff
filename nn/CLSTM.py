import jax
import jax.numpy as jnp
import haiku as hk

from typing import Callable
from jax._src.typing import Array
from haiku._src.recurrent import LSTMState, add_batch
from typing import Any, NamedTuple, Optional, Union
from collections.abc import Sequence

def initial_state(batch_size: Optional[int],
                  input_shape: Sequence[int],
                  output_channels: int) -> LSTMState:
    shape = tuple(input_shape) + (output_channels,)
    state = LSTMState(jnp.zeros(shape), jnp.zeros(shape))
    if batch_size is not None:
        state = add_batch(state, batch_size)
    return (state,)

def get_conv_lstm(output_channels: int=1) -> Callable:
    def conv_lstm_fu(x:Array) -> Array:
        """Defines a vanilla Convolutional LSTM network."""
        time_steps, batch_size, feature_dim1, feature_dim2, feature_dim3 = x.shape

        core = hk.DeepRNN(
            [
                hk.Conv2DLSTM(
                    input_shape=[feature_dim1, feature_dim2],
                    output_channels=output_channels,
                    kernel_shape=[5,5],
                ),
                # jax.nn.relu,
            ]
        )

        # state = core.initial_state(batch_size)
        state = initial_state(batch_size, input_shape=[feature_dim1, feature_dim2],output_channels=output_channels)
        x, state = hk.dynamic_unroll(core, x, state)

        return x, state
    return hk.transform_with_state(conv_lstm_fu)


def get_conv_lstm_layer(output_channels: int=1) -> Callable:
    def conv_lstm_fu(x:Array,
                     state:dict) -> Array:
        """Defines a Convolutional LSTM layer."""
        batch_size, feature_dim1, feature_dim2, feature_dim3 = x.shape

        core = hk.DeepRNN(
            [
                hk.Conv2DLSTM(
                    input_shape=[feature_dim1, feature_dim2],
                    output_channels=output_channels,
                    kernel_shape=[5,5],
                ),
                # jax.nn.relu,
            ]
        )

        y, state = core(x, state)

        return y, state
    # return hk.transform_with_state(conv_lstm_fu)
    return hk.transform(conv_lstm_fu)



if __name__ == "__main__":

    # create random dataset
    key = jax.random.PRNGKey(0)
    train_x = jax.random.normal(key, (5, 3, 10, 8, 2))
    output_channels=4

    conv_lstm = get_conv_lstm(output_channels=output_channels)

    rng_key = jax.random.PRNGKey(1)
    params, state = conv_lstm.init(rng_key, train_x)

    out, state = conv_lstm.apply(params, state, None, train_x)
    assert out[0].shape == (5, 3, 10, 8, output_channels)


    conv_lstm_layer = get_conv_lstm_layer(output_channels=output_channels)

    rng_key = jax.random.PRNGKey(1)
    time_steps, batch_size, feature_dim1, feature_dim2, feature_dim3 = train_x.shape
    state_s = initial_state(batch_size, input_shape=[feature_dim1, feature_dim2],output_channels=output_channels)
    params = conv_lstm_layer.init(rng_key, train_x[0], state_s)

    out, state = conv_lstm_layer.apply(params, None, train_x[0], state_s)
    assert out.shape == (3, 10, 8, output_channels)
    assert state[0][0].shape == (3, 10, 8, output_channels)
    assert state[0][1].shape == (3, 10, 8, output_channels)




# seqs = jnp.arange(3*10*20).reshape(3,10,20)

# def f(seqs):
#     batch_size = seqs.shape[1]
#     deep_rnn = hk.LSTM(hidden_size=5)
    
#     outs, state = hk.dynamic_unroll(deep_rnn, seqs, deep_rnn.initial_state(batch_size))
#     return outs, state 

# ft = hk.transform(f)

# param = ft.init(jax.random.PRNGKey(10), seqs)
# y = ft.apply(param, None, seqs)
# y = ft.apply(param, None, seqs)
