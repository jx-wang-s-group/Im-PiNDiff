import copy
import jax
from typing import Any, Callable, Sequence
from jax import random, numpy as jnp
import flax
from flax import linen as nn

from nn.NN_utils import get_bias_flat2paratree, get_flat2paratree

from nn.siren_nn import act_sin, init_siren_params

from typing import Any, Callable, Sequence, Optional
from jax._src.typing import Array


def activation_fu(activation):
    if activation == 'softplus':
        return jax.nn.softplus
    elif activation == 'leaky_relu':
        return jax.nn.leaky_relu
    elif activation == 'elu':
        return jax.nn.elu
    elif activation == 'gelu':
        return jax.nn.gelu
    elif activation == 'relu':
        return jax.nn.relu
    elif activation == 'sigmoid':
        return jax.nn.sigmoid
    elif activation == 'tanh':
        return jax.nn.tanh
    elif activation == 'sin':
        return act_sin
    else:
        return lambda x:x


class lblock(nn.Module):
  layer: Sequence[int]
  activation: list
  
  def setup(self):
      self.ln = flax.linen.LayerNorm()
      self.net = nn.Sequential([
          nn.Dense(self.layer),
          # self.ln,
          activation_fu(self.activation),
          nn.Dense(self.layer),
      ])
  
  def __call__(self, x):
    #   return self.ln(self.net(x)) + x
      return 1*self.net(x) + 1*x


class MLP(nn.Module):
  layers: Sequence[int]
  activation: list

  def setup(self):
    # self.layers = [nn.Dense(feat) for feat in self.features]
    mlp = [nn.Dense(self.layers[0])]
    for i in range(len(self.layers)-1):
        mlp.append(lblock(self.layers[i], self.activation[0]))
        mlp.append(activation_fu(self.activation[0]))
        mlp.append(nn.Dense(self.layers[i+1]))
    mlp.append(activation_fu(self.activation[1]))
    self.mlp = nn.Sequential(mlp)

  def __call__(self, inputs):
    return self.mlp(inputs)


class MLP_Net():
    def __init__(self, layers, activation=['elu',None], inout_fu = [lambda y: y]*2) -> None:
        self.input_size = layers[0]
        self.inout_fu   = inout_fu
        self.model      = MLP(layers[1:], activation=activation)
        self.siren      = activation[0] == 'sin'
        
    def get_NNparams(self, key):
        key1, key2 = random.split(key, 2)
        representative_input = jnp.ones((self.input_size,))
        NNparams = self.model.init(key1, representative_input)
        if self.siren: NNparams = init_siren_params(key2,NNparams)
        return NNparams

    def __call__(self,NNparams,x):
        x = self.inout_fu[0](x)
        y = self.model.apply(NNparams,x)
        return self.inout_fu[1](y)


class MLP_ParaNet_LastLayer():
    def __init__(self, layers, NF_layers, NFparams_tree, activation=[None,None], inout_fu = [lambda y: y]*2) -> None:
        self.input_size = layers[0]
        self.layers     = layers
        self.NF_layers  = NF_layers
        self.NF_tree    = NFparams_tree
        self.inout_fu   = inout_fu
        self.model      = MLP(layers[1:], activation=activation)
    
    def init_params(self,
                    key:Array,
                    params: dict) -> dict:

        w = params['params']['mlp']['layers_0']['kernel']
        minval, maxval = -jnp.sqrt(6 / w.shape[0]) / 100. , jnp.sqrt(6 / w.shape[0]) / 100.
        key, subkey = random.split(key)
        params['params']['mlp']['layers_0']['kernel'] = random.uniform(subkey, shape=w.shape, minval=minval, maxval=maxval)
        
        NF_tree_flat, _  = jax.flatten_util.ravel_pytree(self.NF_tree)
        NFparams_tree = copy.deepcopy(self.NF_tree)
        NFparams_tree = jax.tree_util.tree_map(lambda x: jnp.ones_like(x)*(len(x.shape)==1), NFparams_tree)
        bias_flat2paratree, nbias = get_bias_flat2paratree(NFparams_tree)
        bias_flat     = jnp.ones(nbias) / nbias
        NF_tree_bias  = bias_flat2paratree(bias_flat)
        NF_bias_flat, _  = jax.flatten_util.ravel_pytree(NF_tree_bias)
        params['params']['mlp']['layers_0']['bias'] = NF_tree_flat + NF_bias_flat
        return params

    def get_NNparams(self, key):
        key1, key2 = random.split(key, 2)
        representative_input = jnp.ones((self.input_size,))
        NNparams = self.model.init(key1, representative_input)
        NNparams = self.init_params(key2,NNparams)
        return NNparams

    def __call__(self,NNparams,x):
        x = self.inout_fu[0](x)
        y = self.model.apply(NNparams,x)
        return self.inout_fu[1](y)


if __name__ == "__main__":

  key1, key2 = random.split(random.key(0), 2)
  x = random.uniform(key1, (8,4))

  model = MLP_Net(layers=[4, 3,4,5])
  NNparams = model.get_NNparams(key2)
  y = model(NNparams, x)

  print('initialized parameter shapes:\n', jax.tree_util.tree_map(jnp.shape, flax.core.unfreeze(NNparams)))
  print('output:\n', y)
