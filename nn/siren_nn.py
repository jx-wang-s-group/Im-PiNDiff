import jax.numpy as jnp
import jax.random as random

from jax._src.typing import Array

w0 = 30

act_sin = lambda x: jnp.sin(w0*x)

def init_siren_params(key:Array,
                      params: dict) -> dict:

    def init_w(key, pytre):
        if isinstance(pytre, dict):
            if 'kernel' in pytre:
                key, subkey = random.split(key)
                w = pytre['kernel']
                minval, maxval = -jnp.sqrt(6 / w.shape[0]) / w0, jnp.sqrt(6 / w.shape[0]) / w0
                pytre['kernel'] = random.uniform(subkey, shape=w.shape, minval=minval, maxval=maxval)
                return key
            else:
                for pytre_key in pytre:
                    key = init_w(key, pytre[pytre_key])
                return key
        else:
            return key

    for i, layers_i in enumerate(params['params']['mlp']):
        pytre = params['params']['mlp'][layers_i]
        key = init_w(key, pytre)

    layers_i = 'layers_0'
    key, subkey = random.split(key)
    w = params['params']['mlp'][layers_i]['kernel']
    minval, maxval = -1 / w.shape[0], 1 / w.shape[0]
    params['params']['mlp'][layers_i]['kernel'] = random.uniform(subkey, shape=w.shape, minval=minval, maxval=maxval)


    # for i, layers_i in enumerate(params['params']['mlp']):
    #     key, subkey = random.split(key)
    #     w = params['params']['mlp'][layers_i]['kernel']
    #     if i == 0:
    #         minval, maxval = -1 / w.shape[0], 1 / w.shape[0]
    #     else:
    #         minval, maxval = -jnp.sqrt(6 / w.shape[0]) / 30, jnp.sqrt(6 / w.shape[0]) / 30
    #     params['params']['mlp'][layers_i]['kernel'] = random.uniform(subkey, shape=w.shape, minval=minval, maxval=maxval)

    return params