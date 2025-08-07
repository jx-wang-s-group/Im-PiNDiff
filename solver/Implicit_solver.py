import jax
import jax.numpy as jnp
import copy
from functools import partial

from solver.boundary import update_BC
from solver.Implicit_solver_utils import *
from nn.adjoint import lin_Adjoint, lin_Adjoint1, Fp_Adjoint1

from math import prod
from jaxopt.linear_solve import solve_bicgstab

from typing import Callable
from jax._src.typing import Array

import matplotlib.pyplot as plt

LargNo = 1e3


def fwd_itr_solver(f:Callable,
                   data:dict,
                   vkey:str = 'phi',
                   tol=1e-16) -> dict:
    """
    Function to compute z_star 
    such that  z_star - f(z_star) = 0
    """
    def cond_fun(carry):
        z_prev, data = carry
        return jnp.linalg.norm(z_prev - data[vkey]) > tol

    def body_fun(carry):
        _, data = carry
        z_prev = data[vkey]
        data[vkey] = f(data)
        # data[key] = 0.4*f(data) + 0.6*data[key]
        return z_prev, data

    phi_pre = data[vkey]
    data[vkey] = f(data)

    init_carry = (phi_pre, data)
    z_star, _ = jax.lax.while_loop(cond_fun, body_fun, init_carry)

    # fig, ax = plt.subplots(1,1, figsize=(3*5, 4))
    # for _ in range(1000):
    #     print(_, jnp.linalg.norm(data[vkey] - f(data)) )
    #     data[vkey] = f(data) 
    #     ax.contourf(data['cell_x'][0], data['cell_x'][1], data[vkey][0])
    #     # plt.colorbar()
    #     plt.savefig('./output/data.png')
    return z_star


def get_Ax(RHS_fu, grid_shape, params, data, vkeys, alpha):
    data_temp = copy.deepcopy(data)
    # @jax.jit
    def Ax_fu(var1D):
        var2D = jax.tree_util.tree_map(lambda v: v.reshape(-1, *grid_shape), var1D)
        data_temp.update(var2D)
        data_temp.update(update_BC(data_temp, vkeys))
        RHS  = RHS_fu(params, data_temp)
        Ax1D = jax.tree_util.tree_map(lambda v2D,rhs: 
                                      (v2D - data['dt']*rhs * alpha).reshape(-1,prod(grid_shape)), 
                                      var2D, RHS)
        return Ax1D
    return Ax_fu

def get_b(RHS_fu, grid_shape, params, data, vkeys, alpha):
    data = update_BC(data, vkeys)
    RHS  = RHS_fu(params, data)
    b = {}
    for vkey in vkeys:
        b[vkey] = (data[vkey] + (1-alpha)* data['dt']*RHS[vkey]).reshape(-1,prod(grid_shape))
    return b

def get_ImTimeStep(args:dict,
                   RHS_fu:Callable,
                   vkeys:list,
                   alpha:float = 0.5,
                   debug:bool = False,
                   tol:float=1e-6) -> Callable:
    """
    Function to step in time, Backward Euler
    """

    if "Burgers1v" in args['case_setup']:
        from jaxopt.linear_solve import solve_bicgstab
        
        _get_Ax = partial(get_Ax, RHS_fu=RHS_fu, grid_shape=args['nCell'], vkeys=vkeys, alpha=alpha)
        _get_b  = partial(get_b , RHS_fu=RHS_fu, grid_shape=args['nCell'], vkeys=vkeys, alpha=alpha)
        
        adjlin = lin_Adjoint1(solve_bicgstab, _get_Ax, _get_b)
        lin_adj = adjlin.add_adjoint_backprop()

        # def PseudoLinSolv(carry, _, data_t, params):
        #     data, ufac = carry
        #     data['vx_pre'] = data['vx']
        #     data['vy_pre'] = data['vy']
        #     data  = update_BC(data, ['vx_pre','vy_pre'])

        #     Ax_fu = get_Ax(RHS_fu, args['nCell'], params, data, vkeys, alpha)
        #     b = get_b(RHS_fu, args['nCell'], params, data, vkeys, alpha, data_t=data_t)

        #     # x, (k,rs)   = solve_bicgstab_custom(Ax_fu, b, init=b, tol=tol)
        #     x, (k,rs)   = solve_bicgstab(Ax_fu, b, init=b, tol=tol, maxiter=1), (0,0)
        #     # x, (k,rs)   = lin_adj(params, data, data_t), (0,0)
        #     x_pre = data['vx']
        #     for vkey in vkeys:
        #         data[vkey] = (1-ufac)*data[vkey] + ufac*x[vkey].reshape([-1, *args['nCell']])
        #     # ufac = ufac*0.8
        #     return (data, ufac), (k,rs, jnp.linalg.norm(x_pre - data['vx'])/prod(args['nCell']))
        
        # def Fixpoint(carry, _, data_t, RHS_t, params):
        #     data, ufac = carry
        #     data['vx_pre'] = data['vx']
        #     data['vy_pre'] = data['vy']
        #     data  = update_BC(data, ['vx_pre','vy_pre'])
            
        #     RHS = RHS_fu(params, data)
        #     x = data_t['vx'] + alpha*data['dt']*RHS['vx'] + (1-alpha)*data['dt']*RHS_t['vx']

        #     x_pre = data['vx']
        #     data['vx'] = (1-ufac)*x_pre + x*ufac
        #     # ufac = ufac*0.8
        #     return (data, ufac), (1,0, jnp.linalg.norm(x_pre - data['vx'])/prod(args['nCell']))
        
        def get_fp(params, data, ufac=1.0):
            # data_t = {'vx':data['vx']}
            data_pret = copy.deepcopy(data)
            data_temp = copy.deepcopy(data)

            data['vx_pre'] = data['vx']
            data['vy_pre'] = data['vy']
            data  = update_BC(data, ['vx_pre','vy_pre'])
            # RHS_t = RHS_fu(params, data)

            def F_fp(params, data, z): 
                data_temp.update({'vx':z['vx']})
                data_temp.update({'vy':z['vx']})
                data_temp.update({'vx_pre':z['vx']})
                data_temp.update({'vy_pre':z['vx']})
                data_temp.update(update_BC(data_temp, ['vx','vy','vx_pre','vy_pre']))
                
                # RHS = RHS_fu(params, data_temp)
                # z_new = data_t['vx'] + alpha*data_temp['dt']*RHS['vx'] + (1-alpha)*data_temp['dt']*RHS_t['vx']

                # Ax_fu = get_Ax(RHS_fu, args['nCell'], params, data_temp, vkeys, alpha)
                # b = get_b(RHS_fu, args['nCell'], params, data_temp, vkeys, alpha, data_t=data_t)
                # z_new = solve_bicgstab(Ax_fu, b, init=b, tol=tol, maxiter=10)
                z_new = lin_adj(params, data_temp, data_pret)
                z_new = z_new['vx'].reshape(-1,*args['nCell'])

                return {'vx':(1-ufac)*z['vx'] + ufac*z_new}
            return F_fp


        def nonLinSolver(params, data):
            data_t = {'vx':data['vx']}
            RHS_t  = RHS_fu(params, data)

            # ufac = 1.0
            # for _ in range(50):
            #     (data, ufac), (k,rs, ro) = PseudoLinSolv((data, ufac), None, data_t, params)
            #     # print (_, ufac, ro,  k,rs)
            # sol_info  = {'iter':k,'res':ro}

            F_fp = get_fp(params, data, 0.7)
            f_z = lambda z: F_fp(params, data, z)
            z = {'vx':data['vx']}
            for _ in range(10):
                z_pre = z
                z = f_z(z)
            data['vx'], (k,rs,ro) = z['vx'], (1,0,[jnp.linalg.norm(z_pre['vx'] - z['vx'])/prod(args['nCell'])])

            # (data,_), (k,rs,ro) = jax.lax.scan(partial(Fixpoint, data_t=data_t, RHS_t=RHS_t, params=params), init=(data, 0.7), xs=None, length=30)
            # (data,_), (k,rs,ro) = jax.lax.scan(partial(PseudoLinSolv, data_t=data_t, params=params), init=(data, 0.7), xs=None, length=30)
            sol_info  = {'iter':jnp.max(k),'res':ro[-1]}
            # print (ro[-1],  jnp.max(k),jnp.max(rs))

            # def cond_fun(carry):
            #     _, (k,rs, ko,ro) = carry
            #     return ro > 1e-6

            # def body_fun(carry):
            #     (data, ufac), (k,rs, ko,ro)= carry
            #     (data, ufac), (k,rs, ro) = PseudoLinSolv((data, ufac), None, data_t, params)
            #     return (data, ufac), (k,rs, ko+1,ro)

            # init_carry = ((data, 1.0), (0,0,0,10.))
            # (data, _), (k,rs, ko,ro) = jax.lax.while_loop(cond_fun, body_fun, init_carry)
            # sol_info  = {'iter':ko,'res':ro}

            return data, sol_info
        
        adjFp = Fp_Adjoint1(get_fp, vkeys, length=20, lsolver='fwd')
        # Fp_adj = adjFp.add_adjoint_backprop()
        Fp_adj = adjFp.fp_solver_fwd

    else:
        from solver.custom_linear_solve import solve_bicgstab
        
        _get_Ax = partial(get_Ax, RHS_fu=RHS_fu, grid_shape=args['nCell'], vkeys=vkeys, alpha=alpha)
        _get_b  = partial(get_b , RHS_fu=RHS_fu, grid_shape=args['nCell'], vkeys=vkeys, alpha=alpha)
        
        adj = lin_Adjoint(solve_bicgstab, _get_Ax, _get_b)
        Adj_ImTimestep = adj.add_adjoint_backprop()
        # Adj_ImTimestep = adj.linear_solve_fwd

    def ImTimeStep(params:dict,
                   data:dict) -> dict:
        # phi_pre = data[vkey]
        # f = lambda d: phi_pre + d['dt']*(RHS_fu(d)+RHS_fu(data))*0.5
        # data[key] = fwd_itr_solver(f, data, tol=1e-10)
        
        
        if "Burgers1v" in args['case_setup']:

            # data, sol_info = nonLinSolver(params, data)

            x, sol_info = Fp_adj(params, data), {'iter':0,'res':0}
            
            for vkey in vkeys:
                data[vkey] = x[vkey].reshape([-1, *args['nCell']])

        else:
            # Ax_fu = get_Ax(RHS_fu, args['nCell'], params, data, vkeys, alpha)
            # b     = get_b (RHS_fu, args['nCell'], params, data, vkeys, alpha)
            # x, (k,rs)   = solve_bicgstab_custom(Ax_fu, b, init=b, tol=tol)
            # x, (k,rs)   = solve_bicgstab(Ax_fu, b, init=b, tol=tol), (0,0)

            x, sol_info = Adj_ImTimestep(params, data)#, (0,0)
            # sol_info    = {'iter':k,'res':rs}
            # jax.debug.print('info = {r}, {n}', r = sol_info['error'], n = sol_info['niter'])

            for vkey in vkeys:
                data[vkey] = x[vkey].reshape([-1, *args['nCell']])
        
        return data, sol_info

    return jax.jit(ImTimeStep) if not debug else ImTimeStep
