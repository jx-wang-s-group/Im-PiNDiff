import jax.numpy as jnp

from utils.utils import grad_phi_face_fu
# from solver.boundary import update_BC
from utils.utils import face_value_fu

from typing import Callable
from jax._src.typing import Array

def get_diff_fu(args:dict,
                vkey:str) -> Callable:
    """
    Function to calculate diffusion term
    """
    def diff_fu(params:dict,
                data:dict) -> Array:
        kdd_x   = face_value_fu(data, 'kx', scheme='central') # b*nx*ny --> b*nf*nx*ny
        kdd_y   = face_value_fu(data, 'ky', scheme='central') # b*nx*ny --> b*nf*nx*ny
        kdd     = jnp.stack([kdd_x, kdd_y], axis=2) # b*nf*nx*ny --> b*nf*nd*nx*ny 

        grad_phi_face = grad_phi_face_fu(data, vkey) # b*nx*ny --> b*nf*nd*nx*ny
        face_A  = jnp.repeat(data['face_A'][None], args['nBatch'], axis=0) # nf*nd*nx*ny --> b*nf*nd*nx*ny
        gphi_A  = grad_phi_face * face_A # b*nf*nd*nx*ny x b*nf*nd*nx*ny --> b*nf*nd*nx*ny
        kgphi_A = jnp.sum(gphi_A* kdd, axis=2)# b*nf*nd*nx*ny x b*nf*nd*nx*ny --> b*nf*nx*ny
        diff    = jnp.sum(kgphi_A, axis=1) # b*nf*nx*ny --> b*nx*ny
        return diff
    
    return diff_fu

