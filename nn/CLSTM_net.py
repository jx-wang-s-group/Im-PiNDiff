import jax
import jax.numpy as jnp
import optax
from functools import partial

import matplotlib.pyplot as plt

from haiku._src.recurrent import LSTMState
from nn.CLSTM import initial_state
from nn.adjoint import Fp_Adjoint

from typing import Callable
from jax._src.typing import Array

ip_len = 1


def get_Model(net:Callable,
              train_len:int,
              TeacherForcing:bool = True) -> Callable:
    debug = False

    def f_rhs(params, x, state):
        state = (LSTMState(jnp.zeros_like(x[...,0:1]), x[...,0:1]),)
        y, state = net.apply(params, None, x, state)
        return state
    
    adj = Fp_Adjoint(f_rhs, length=10, lsolver='fwdc', tol=[1e-6,1e-6])
    iter_CLSTM = adj.get_fp_layer()

    def Model(params:dict,
              state:dict, 
              datat:dict,
              tidxs:Array,
              **args) -> Array:
        
        # time_steps, batch_size, feature_dim1, feature_dim2 = datat['phi'].shape
        # state_s = initial_state(batch_size, input_shape=[feature_dim1, feature_dim2],output_channels=1)
        state_0 = (LSTMState(jnp.zeros_like(datat['phi'][0,...,None]), datat['phi'][0,...,None]),)

        if TeacherForcing:
            def step_data(state, tidx):
                x = datat['phi'][tidx[0],:,:,:]
                x = jnp.stack([x, (datat['tarr'][tidx[1]]-datat['tarr'][tidx[0]])*jnp.ones_like(x)], axis=-1)
                state = iter_CLSTM(params, x, state)
                return state, state[0][1][...,0]
            state, out = jax.lax.scan(step_data, init=state_0, xs=tidxs)
        
        else:
            def step_data(x, tidx, tidx0):
                state = (LSTMState(jnp.zeros_like(x[...,None]), x[...,None]),)
                x = jnp.stack([x, (datat['tarr'][tidx]-datat['tarr'][tidx0])*jnp.ones_like(x)], axis=-1)
                state = iter_CLSTM(params, x, state)
                x = state[0][1][...,0]
                return x, x
            state_h, out = jax.lax.scan(partial(step_data, tidx0=tidxs[0]), init=datat['phi'][0], xs=tidxs)

        return out
    
    return jax.jit(Model) if not debug else Model
 
# def get_Model(net:Callable) -> Callable:
#     def Model(params:dict,
#               state:dict, 
#               datat:dict) -> Array:
        
#         x = datat['phi']
#         time_steps, batch_size, feature_dim1, feature_dim2 = x.shape
#         out = []
#         for t in range(time_steps-ip_len):
#             # y, state = net.apply(params, state, None, x[t:t+5,:,:,:,None])
#             state_s = initial_state(batch_size, input_shape=[feature_dim1, feature_dim2],output_channels=1)
#             y, state = net.apply(params, state, None, x[t:t+ip_len,:,:,:,None], state_s, length=ip_len)
#             out.append(y[0][-1,:,:,:,0])
#         return jnp.stack(out)
    
#     return Model
 
def get_nnl_fu(Model:Callable) -> Callable:

    def nll_fu(params:dict,
               state:dict,
               datat:dict,
               **args) -> Array:
        time_steps = datat['phi'].shape[0]
        tidxs = jnp.stack([jnp.arange(time_steps-1), jnp.arange(1,time_steps)], axis=-1)
        pred  = Model(params, state, datat, tidxs, **args)
        loss  = jnp.mean(jnp.square(pred-datat['phi'][tidxs[:,1]]))

        tidxs = jnp.stack([jnp.arange(time_steps-5), jnp.arange(5,time_steps)], axis=-1)
        pred  = Model(params, state, datat, tidxs, **args)
        loss += jnp.mean(jnp.square(pred-datat['phi'][tidxs[:,1]]))

        tidxs = jnp.stack([jnp.arange(time_steps-10), jnp.arange(10,time_steps)], axis=-1)
        pred  = Model(params, state, datat, tidxs, **args)
        loss += jnp.mean(jnp.square(pred-datat['phi'][tidxs[:,1]]))

        tidxs = jnp.stack([jnp.zeros(time_steps-1, dtype=int), jnp.arange(1,time_steps)], axis=-1)
        pred  = Model(params, state, datat, tidxs, **args)
        loss += jnp.mean(jnp.square(pred-datat['phi'][tidxs[:,1]]))

        for w in jax.tree_util.tree_leaves(params):
            loss += 1e-7*jnp.mean(jnp.square(w))

        return loss

    return nll_fu


def get_update_fu(opt:Callable, 
                  nll_fu:Callable, 
                  debug:bool = False) -> Callable:
    
    def update_fu(params:dict,
                  state:dict,
                  datat:dict, 
                  opt_state,
                  **args):
        if debug:
            loss = nll_fu(params, state, datat**args)

        nll_out, grads  = jax.value_and_grad(nll_fu)(params, state, datat, **args)
        
        update, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, update)


        # for w in jax.tree_util.tree_leaves(params):
        #     # print(w.shape)
        #     # wd= jnp.diag(w)
        #     w= jnp.clip(w, a_min=-1, a_max=1)
        #     # w = (1-jnp.eye())w + jnp.diag(wd)

        return params, opt_state, nll_out
    return jax.jit(update_fu) if not debug else update_fu


def train(net, params,state,datat,opt,opt_state, nepochs):

    time_steps = datat['phi'].shape[0]

    # Model = get_Model(net)
    # nll_fu = get_nnl_fu(Model)
    # update_fu = get_update_fu(opt, nll_fu, debug = False)
    Model_ro = get_Model(net, 0, TeacherForcing=False)

    loss_list = []
    for epoch in range(nepochs):

        if epoch%500 == 0:
            train_len = min(1+int(epoch/500), 11)
            Model = get_Model(net, train_len)
            nll_fu = get_nnl_fu(Model)
            update_fu = get_update_fu(opt, nll_fu, debug = False)

        params, opt_state, loss = update_fu(params,state,datat,opt_state)
        loss_list.append(loss)

        if epoch%10 == 0:
            print(epoch, loss)
            fig, ax = plt.subplots(5,3, figsize=(3*15, 5*4))
            # tidxs = jnp.stack([jnp.zeros(time_steps-1, dtype=int), jnp.arange(1,time_steps)], axis=-1)
            tidxs = jnp.arange(1,time_steps)
            pred = Model_ro(params, state, datat, tidxs)
            c = 0
            for i in range(5):
                t = (len(pred) // 5) * i
                ax[i,0].contourf(datat['cell_x'][0], datat['cell_x'][1], pred[t,c])
                ax[i,1].contourf(datat['cell_x'][0], datat['cell_x'][1], datat['phi'][1:][t,c])
                ax[i,2].contourf(datat['cell_x'][0], datat['cell_x'][1], pred[t,c]-datat['phi'][1:][t,c])
            plt.savefig('./output/pred.png')

    return params, loss_list
