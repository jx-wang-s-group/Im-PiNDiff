import jax, time
import copy
import jax.numpy as jnp
import optax
from functools import partial
from math import prod

import matplotlib.pyplot as plt

# from nn.adjoint import Fp_Adjoint
from solver.diff_eq_solver import get_roleout
from utils.plots_pred import plot_pred, plot_pred_gif
from utils.plots_pred import plot_1D_wtime
from utils.utils import PyTree

from typing import Callable
from jax._src.typing import Array



def get_nnl_fu(args:dict,
               roleout:Callable,
               datat_label:dict,
               vkeys_train:list,
               traintime:Array,
               trainbatch:Array) -> Callable:
    if traintime == None:
        raise ValueError('provide traintime')
    testnbatch = jnp.setdiff1d(jnp.arange(args['nBatch']), trainbatch)

    def nll_fu(params:dict,
               data_ICBC:dict,
               **vargs) -> Array:
        data, sol_info = roleout(params, data_ICBC, **vargs)
        for vkey in vkeys_train:
            loss  = jnp.mean(jnp.square(data['datat'][vkey][traintime,][:,trainbatch,]-datat_label['datat'][vkey][traintime,][:,trainbatch,]) ) / jnp.mean(datat_label['datat'][vkey][traintime,][:,trainbatch,])
            val_loss  = jnp.mean(jnp.square(data['datat'][vkey][traintime,][:,testnbatch,]-datat_label['datat'][vkey][traintime,][:,testnbatch,]) ) / jnp.mean(datat_label['datat'][vkey][traintime,][:,testnbatch,])
            # loss  = jnp.mean(jnp.square(data['datat'][vkey][:]-datat_label['datat'][vkey][:]))

        # for vkey in args['train']:
        #     diff = (data['datat'][vkey]-datat_label['datat'][vkey]).reshape(*data['datat'][vkey].shape[:2],-1) / jnp.mean(datat_label['datat'][vkey])
        #     loss += jnp.mean(jnp.square(diff[:,:,idx])) * nSensor/prod(args['nCell'])

        # for w in jax.tree_util.tree_leaves(params):
        #     loss += jnp.mean(jnp.square(w))
        loss += 1e-4*jnp.mean(jnp.square(jax.flatten_util.ravel_pytree(params)[0]))

        return loss, (val_loss, sol_info)

    return nll_fu


def get_update_fu(opt_fu:Callable, 
                  nll_fu:Callable,
                  mask:dict, 
                  debug:bool = False) -> Callable:
    
    def update_fu(params:dict,
                  data_ICBC:dict, 
                  opt_state,
                  **vargs):
        # if debug:
        #     loss = nll_fu(params, data_ICBC,**vargs)

        (nll_out, (val_loss, sol_info)), grads  = jax.value_and_grad(nll_fu, has_aux=True)(params, data_ICBC, **vargs)

        # grads = jax.tree_util.tree_map(lambda g,m: g*m, grads, mask)
        grad_norm = jnp.sqrt(sum([jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(grads)]))
        
        for k in params:
            update, opt_state[k] = opt_fu[k].update(grads[k], opt_state[k])
            params[k] = optax.apply_updates(params[k], update)

        return params, opt_state, nll_out, grad_norm, val_loss, sol_info
    return jax.jit(update_fu) if not debug else update_fu


def train(args:dict, 
          trainables:(Callable, Callable, dict, dict, dict), 
          datat_label:dict,
          **kwargs) -> tuple:
    print('=================== Traiing ==========================')

    models, opt_fu, params, mask, opt_state = trainables
    data_ICBC = copy.deepcopy(datat_label)
    data_ICBC['dt'] = args['dt']    #run with specified dt
    del data_ICBC['datat']
    data_ICBC['tcur'] = jnp.array(0)
    for vkey in args['state_var']:
        data_ICBC.update({vkey:datat_label['datat'][vkey][0]})
    
    roleout   = get_roleout(args, models, sim_tarr=args['sim_tarr'], debug=args['debug'])
    nll_fu    = get_nnl_fu(args, roleout, datat_label, args['state_var'], traintime=kwargs['traintime'], trainbatch=kwargs['trainbatch'])
    update_fu = get_update_fu(opt_fu, nll_fu, mask, debug=args['debug'])

    if args['epochstart']:
        params = PyTree.load(args['path']+'/checkpoints', name='params_'+str(args['epochstart']))

    params, opt_state, loss, grad_norm, val_loss, sol_info = update_fu(params,data_ICBC,opt_state)
    
    starttime = time.time()
    train_loss_list, val_loss_list, gradn_list = [], [], []
    Im_sol_info = []
    for epoch in range(args['epochstart'], args['nepochs']+1):

        params, opt_state, loss, grad_norm, val_loss, sol_info = update_fu(params,data_ICBC,opt_state)
        train_loss_list.append(loss)
        val_loss_list.append(val_loss)
        gradn_list.append(grad_norm)
        Im_sol_info.append(sol_info)

        if epoch%args['plot_depoch'] == 0:
            print(epoch, loss)#, jnp.mean(data['ky']), params['condel'])
            data, sol_info = roleout(params, data_ICBC)
            for k in args['trac_var']:
                plot_pred(args, data, datat_label, vkeys=[k], case = 0)
    endtime = time.time()
    print("Train time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))
    
    data['data_label'] = datat_label['datat']
    data['traintime'] = kwargs['traintime']
    PyTree.save(params, args['path']+'/checkpoints', name='params_'+str(args['nepochs']))
    PyTree.save(data, args['path']+'/checkpoints', name='data_pred')
    PyTree.save(train_loss_list, args['path'], name='train_loss')
    PyTree.save(val_loss_list, args['path'], name='val_loss')
    PyTree.save(gradn_list, args['path'], name='grad_norm')
    PyTree.save(Im_sol_info, args['path'], name='Im_sol_info')

    # plot_1D_wtime( args, data, datat_label, vkey=args['trac_var'],nsample_times=5, case = -1)
    # plot_pred_gif(args, data, datat_label, vkey=args['trac_var'], case = -1)
    # plot_pred_gif(args, data, datat_label, vkey='vy', case = -1)

    return params, train_loss_list
