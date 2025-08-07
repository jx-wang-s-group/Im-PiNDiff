import jax
import jax.numpy as jnp

from solver.convection import get_conv_fu
from solver.diffusion import get_diff_fu
from solver.Explicit_solver import get_FwdEuler, get_rk4
from solver.Implicit_solver import get_ImTimeStep
from solver.boundary import update_BC

from nn.CondNF_nn import CNFnn

from typing import Callable
from jax._src.typing import Array

def get_RHS_fu(RHS_fu_dict:dict) -> Callable:
    """
    Function to compute the RHS
    """
    def RHS_fu(params:dict,
               data:dict) -> Array:
        phi_dot = {vkey:0 for vkey in RHS_fu_dict.keys()}
        for  vkey in RHS_fu_dict:
            for  fui in RHS_fu_dict[vkey]:
                phi_dot[vkey] += RHS_fu_dict[vkey][fui](params, data)
            phi_dot[vkey] = phi_dot[vkey] / data['cell_vol']
        return phi_dot
    
    return RHS_fu


def get_time_step(args: dict, 
                  debug: bool = False) -> Callable:
    # debug = False
    RHS_fu_dict = {vkey:{} for vkey in args['state_var']}
    for vkey in args['state_var']:
        RHS_fu_dict[vkey]['conv_fu'] = get_conv_fu(args, vkey=vkey)
        RHS_fu_dict[vkey]['diff_fu'] = get_diff_fu(args, vkey=vkey)
    RHS_fu = get_RHS_fu(RHS_fu_dict)

    if 'rk4' in args['odesolve']:
        time_step = get_rk4(args, RHS_fu, vkeys=args['state_var'], CFL=0.1, debug=debug)
    elif 'FwdEuler' in args['odesolve']:
        time_step = get_FwdEuler(args, RHS_fu, vkeys=args['state_var'], CFL=0.5, debug=debug)
    elif 'CrankNicolson' in args['odesolve']:
        time_step = get_ImTimeStep(args, RHS_fu, vkeys=args['state_var'], alpha=0.5, debug=debug)
    elif 'BackEuler' in args['odesolve']:
        time_step = get_ImTimeStep(args, RHS_fu, vkeys=args['state_var'], alpha=1.0, debug=debug)
    return time_step
    

def update_adv_diff_val(args: dict,
                        models: Callable) -> Callable:
    # fac =  {'vx': 1., 'vy': 1., 'kx': 1e-2, 'ky': 1e-2}
    # fu = {  'vx': lambda p, d: args['vel_fu'][0](d),  
    #         'vy': lambda p, d: args['vel_fu'][1](d),
    #         'kx': lambda p, d: args['diff_fu'][0](d),
    #         'ky': lambda p, d: args['diff_fu'][1](d),}
    
    fu = {}
    fu['kx'] = lambda p, d: args['diff_fu'][0](d)
    fu['ky'] = lambda p, d: args['diff_fu'][1](d)
    # fu['ky'] = lambda p, d: jnp.abs(p['condel'])*jnp.ones_like(d['ky'])
    if "AdvDiff" in args['case_setup']:
        fu['vx'] = lambda p, d: args['vel_fu'][0](d)
        fu['vy'] = lambda p, d: args['vel_fu'][1](d)
    
    if not args['gen_data']:
        for vkey in args['train']:                 ##################### WRONG ##################################
            fu[vkey] = CNFnn(args, models, vkey)

    # only one k=kx=ky
    fu['ky']=fu['kx']

    def update_adv_diff(params, data, **vargs):
            
        for vkey in fu:
            data[vkey]  = fu[vkey](params, data)*args['ord'][vkey]

        for vkey in ['kx','ky']:
            data[vkey] = jnp.clip(data[vkey], a_min=1e-3)

        data = update_BC(data, ['vx','vy'])
        data = update_BC(data, ['kx','ky'], bc_type = 'gradFree')

        return data 
    
    return update_adv_diff
    


def get_roleout(args:dict, 
                models:Callable,
                sim_tarr:Array,
                debug:bool = False) -> Callable:
    # debug = True
    
    update_adv_diff = update_adv_diff_val(args, models)

    time_step = get_time_step(args, debug=debug)

    def roleout_step(params, data):

        # time step
        data, sol_info_ = time_step(params, data)
        data['tcur'] += data['dt']

        if "Burgers1v" in args['case_setup']:
            data['vy'] = data['vx']
            data = update_BC(data, vkeys=['vy'])

        # Boundary condition.
        data = update_BC(data, vkeys=args['state_var'])

        return data, sol_info_


    if debug:
        def roleout(params, data):

            datat = {'tarr':[]}
            for vkey in args['trac_var']:
                datat[vkey] = []
            sol_info = {'error':0., 'niter':0.}
            sol_info_ = {'error':0., 'niter':0.}
            
            pre_t=0
            for iter, t in enumerate(sim_tarr[1:]):
                data['dt'] = t-pre_t
                # update values predicted by nn
                data = update_adv_diff(params, data)

                # store
                store = {'tarr':data['tcur'][None]} 
                for vkey in args['trac_var']:
                    data[vkey] = jnp.clip(data[vkey], a_max= 100, a_min=-100)
                    store[vkey] = data[vkey][None]
                
                for Var in datat.keys():
                    datat[Var].append(store[Var])

                data, sol_info_ = roleout_step(params, data)
                pre_t = t

                for Var in sol_info_.keys():
                    sol_info[Var] = jnp.maximum(sol_info[Var], sol_info_[Var])

                if args['gen_data']:
                    print(iter,'t=', data['tcur'],'dt=', data['dt'], 'sol=', sol_info_['iter'], sol_info_['res'])

            # store
            store = {'tarr':data['tcur'][None]} 
            for vkey in args['trac_var']:
                store[vkey] = data[vkey][None]
            
            for Var in datat.keys():
                datat[Var].append(store[Var])

            for Var in datat.keys():
                datat[Var] = jnp.concatenate(datat[Var], axis=0)

            data['datat'] = datat
            return data, sol_info

    else:
        def roleout(params, data):

            # @jax.checkpoint
            def step(carry, t):
                data, pre_t = carry
                data['dt'] = t-pre_t
                
                # update values predicted by nn
                data = update_adv_diff(params, data)

                # store
                store = {'tarr':data['tcur']} 
                for vkey in args['trac_var']:
                    data[vkey] = jnp.clip(data[vkey], a_max= 100, a_min=-100)
                    store[vkey] = data[vkey]
                
                data, sol_info_ = roleout_step(params, data)
                
                # for Var in sol_info_.keys():
                #     sol_info[Var] = jnp.maximum(sol_info[Var], sol_info_[Var])

                carry = data, t
                store['info'] = sol_info_

                return carry, store
            
            # sol_info = {'error':0., 'niter':0.}
            # (data, sol_info, _), datat = jax.lax.scan(step, init=(data, sol_info, 0), xs=None, length=length+1)
            (data, _), datat = jax.lax.scan(step, init=(data, 0), xs=sim_tarr[1:])
            sol_info = datat['info']

            # store
            datat['tarr'] = jnp.concatenate([datat['tarr'], data['tcur'][None]])
            for vkey in args['trac_var']:
                datat[vkey] = jnp.concatenate([datat[vkey], data[vkey][None]])
                
            data['datat'] = datat
            return data, sol_info
   

    return jax.jit(roleout) if not debug else roleout
