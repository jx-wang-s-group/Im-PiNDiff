import jax, copy
import jax.numpy as jnp

from typing import Callable
from jax._src.typing import Array

def dt_fu(data:Callable,
          CFL:float = 0.5):
    
    vel = jnp.stack([data['vx'],data['vy']], axis=1)
    difcoef = jnp.stack([data['kx'],data['ky']], axis=1)

    dx  = jnp.minimum(jnp.max(data['cell_dx'][:,0], axis=0), 
                        jnp.max(data['cell_dx'][:,1], axis=0))
    dtc = CFL * dx/jnp.linalg.norm(jnp.max(jnp.abs(vel), axis=0), axis=0)
    dtd = CFL * dx*dx/(2*jnp.max(difcoef, axis=(0,1)))
    dt  = jnp.minimum(dtc, dtd)
    return dt

def get_FwdEuler(args:dict,
                 RHS_fu:Callable, 
                 vkeys:list,
                 CFL:float = 0.5, 
                 debug:bool = False) -> Callable:
    """
    Function to step in time, Forward Euler
    """
    def FrdEuler(params:dict,
                 data:dict) -> dict:
        # data['dt'] = jnp.min(dt_fu(data, CFL))

        k1 = RHS_fu(params, data)
        for vkey in vkeys:
            data[vkey] = data[vkey] + data['dt'] * k1[vkey]
        return data, {'iter':0., 'res':0.}

    return jax.jit(FrdEuler) if not debug else FrdEuler


def get_rk4(args:dict,
            RHS_fu:Callable, 
            vkeys:str,
            CFL:float = 0.5, 
            debug:bool = False) -> Callable:
    """
    Function to step in time, Runge-kutta 4th order
    """
    def rk4(params:dict,
            data:dict) -> dict:
        # h = jnp.min(dt_fu(data, CFL))
        h = data['dt']
        phi0 = dict()
        for vkey in vkeys:
            phi0[vkey] = copy.deepcopy(data[vkey])

        k1 = RHS_fu(params, data)
        for vkey in vkeys:
            data[vkey] += h * k1[vkey] / 2
        k2 = RHS_fu(params, data)
        for vkey in vkeys:
            data[vkey] += h * k2[vkey] / 2
        k3 = RHS_fu(params, data)
        for vkey in vkeys:
            data[vkey] += h * k3[vkey]
        k4 = RHS_fu(params, data)
        for vkey in vkeys:
            data[vkey] = phi0[vkey] +  1.0 / 6.0 * h * (k1[vkey] + 2 * k2[vkey] + 2 * k3[vkey] + k4[vkey])

        return data, {'error':0., 'niter':0.}

    return jax.jit(rk4) if not debug else rk4
