import jax.numpy as jnp
from math import prod

from nn.NN_utils import get_bias_flat2paratree, get_flat2paratree

from typing import Any, Callable
from jax._src.typing import Array

class CNFnn():

    def __init__(self,
                 args:dict,
                 models: Callable,
                 vkey: str) -> None:
        self.args = args
        self.models = models
        self.vkey = vkey
        # self.bias_flat2paratree, _ = get_bias_flat2paratree(args['NFparams_tree'])
        self.flat2paratree, _      = get_flat2paratree(args['NFparams_tree'][vkey])
        self.dynamic = 0 if 'steady' in args['case_setup'] else 1


    def __call__(self, params:dict,
                 data:dict) -> Array:
        HYmodel,  PJmodel,  NFmodel  = self.models['NF']
        HYparams, PJparams, NFparams = params['NF']

        t = data['tcur'][None] * self.dynamic
        hidd_vec = HYmodel(HYparams, t)
        proj_vec = PJmodel[self.vkey](PJparams[self.vkey], hidd_vec)

        # bias_params = bias_flat2paratree(proj_vec)
        # NFparams_mod = jax.tree_util.tree_map(lambda xp, tp: xp+tp, bias_params, NFparams)

        wb_params = self.flat2paratree(proj_vec)

        x = jnp.transpose(data['cell_x'], (1,2,0))
        out = jnp.transpose(NFmodel[self.vkey](wb_params, x), (2,0,1))
        out = jnp.stack([out[0]]*self.args['nBatch'])

        return out
