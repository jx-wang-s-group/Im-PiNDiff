import jax.numpy as jnp

# from solver.boundary import update_BC
from utils.utils import face_value_fu

from typing import Callable
from jax._src.typing import Array

def get_conv_fu(args:dict,
                vkey:str,
                scheme:str='central') -> Callable:
    """
    Function to calculate convection term
    """
    scheme='upwind'

    def conv_fu(params:dict,
                data:dict) -> Array:
        face_velx= face_value_fu(data, 'vx', scheme=scheme) # b*nx*ny --> b*nf*nx*ny
        face_vely= face_value_fu(data, 'vy', scheme=scheme) # b*nx*ny --> b*nf*nx*ny
        # face_vel = jnp.stack([face_velx, face_vely], axis=2) # b*nf*nx*ny --> b*nf*nd*nx*ny 
        face_vel = jnp.stack([face_velx, face_vely], axis=2) # b*nf*nx*ny --> b*nf*nd*nx*ny 
        face_A   = jnp.repeat(data['face_A'][None], face_vel.shape[0], axis=0) # nf*nd*nx*ny --> b*nf*nd*nx*ny
        vel_area = jnp.sum(face_vel*face_A, axis=2) # b*nf*nd*nx*ny x b*nf*nd*nx*ny --> b*nf*nx*ny
        phi_face = face_value_fu(data, vkey, scheme=scheme) # phi: b*nx*ny --> b*nf*nx*ny
        conv     = jnp.sum(vel_area*phi_face, axis=1) #  b*nf*nx*ny x b*nf*nx*ny --> b*nx*ny
        return - conv
    
    def conv_fu_Burgers(params:dict,
                        data:dict) -> Array:
        conv = conv_fu(params, data)
        return 0.5* conv
    
    def conv_fu_lin_for_Burgers(params:dict,
                                data:dict) -> Array:
        face_velx= face_value_fu(data, 'vx_pre', scheme='central') # b*nx*ny --> b*nf*nx*ny
     #    face_vely= face_value_fu(data, 'vy_pre', scheme='central') # b*nx*ny --> b*nf*nx*ny
        # face_vel = jnp.stack([face_velx, face_vely], axis=2) # b*nf*nx*ny --> b*nf*nd*nx*ny 
        face_vel = jnp.stack([face_velx, face_velx], axis=2) # b*nf*nx*ny --> b*nf*nd*nx*ny   ######################## WRONG ####################
        face_A   = jnp.repeat(data['face_A'][None], face_vel.shape[0], axis=0) # nf*nd*nx*ny --> b*nf*nd*nx*ny
        vel_area = jnp.sum(face_vel*face_A, axis=2) # b*nf*nd*nx*ny x b*nf*nd*nx*ny --> b*nf*nx*ny
        phi_face = face_value_fu(data, 'vx', scheme='central') # phi: b*nx*ny --> b*nf*nx*ny
        conv     = jnp.sum(vel_area*phi_face, axis=1) #  b*nf*nx*ny x b*nf*nx*ny --> b*nx*ny
        return - 0.5* conv
    
    if "Burgers1v" in args['case_setup']:
         if 'FwdEuler' in args['odesolve'] or 'rk4' in args['odesolve']:
              scheme='central'
              return conv_fu_Burgers
         else:
              return conv_fu_lin_for_Burgers
    else:
         return conv_fu
    
