import jax
import jax.numpy as jnp
import pickle

from jax._src.typing import Array

class PyTree():
    
    @staticmethod
    def set_val(pytree, val):
        return jax.tree_util.tree_map(lambda x: val, pytree)

    @staticmethod
    def extract(pytreeb, n):
        return jax.tree_util.tree_map(lambda x: x[n], pytreeb)

    @staticmethod
    def extract_all(pytreeb):
        l = len(jax.tree_util.tree_leaves(pytreeb)[0])
        return [jax.tree_util.tree_map(lambda x: x[i], pytreeb) for i in range(l)], l

    @staticmethod
    def combine(pytree):
        pytreeb   = jax.tree_util.tree_map(lambda x: jax.lax.expand_dims(x,[0]), pytree[0])
        for nM in range(1,len(pytree)):
            pytreeb   = jax.tree_util.tree_map(lambda x,y: jnp.concatenate((x,jax.lax.expand_dims(y,[0])), axis=0), pytreeb, pytree[nM])
        return pytreeb

    @staticmethod
    def combine_copy(pytree, l):
        pytreeb   = jax.tree_util.tree_map(lambda x: jax.lax.expand_dims(x,[0]), pytree)
        for _ in range(1,l):
            pytreeb   = jax.tree_util.tree_map(lambda x,y: jnp.concatenate((x,jax.lax.expand_dims(y,[0])), axis=0), pytreeb, pytree)
        return pytreeb

    @staticmethod
    def random_split_like_tree(rng_key, pytree, treedef=None):
        if treedef is None:
            treedef = jax.tree_structure(pytree)
        keys = jax.random.split(rng_key, treedef.num_leaves)
        return jax.tree_unflatten(treedef, keys)
        
    @staticmethod
    def save(pytree, path, name):
        print("Saving "+name+" pytree")
        path = path+'/'+name+'_pytree.hdf5'
        with open(path, 'wb') as file:
            pickle.dump(pytree, file)

    @staticmethod
    def load(path, name):
        print("loading "+name+" parameters")
        try:
            path = path+'/'+name+'_pytree.hdf5'
            with open(path, 'rb') as file:
                pytree = pickle.load(file)
            print('Found '+name+' parameters')
        except FileNotFoundError:
            print('Error: Could not find parameters')
            return
        return  pytree


def dummy_scan_fu(f, init, length):

    def scan_fu(x,_, **args):
        return f(x, **args) , None
    
    return jax.lax.scan(scan_fu, init=init, xs=None, length=length)[0]


###################### WRONG #################################

def stack_bc(phi:Array, phi_bc:list, trim:bool) -> list:
    phi_l = jnp.concatenate((phi_bc[0][...,None,:], phi), axis=-2)
    phi_r = jnp.concatenate((phi, phi_bc[1][...,None,:]), axis=-2)
    phi_b = jnp.concatenate((phi_bc[2][...,:,None], phi), axis=-1)
    phi_t = jnp.concatenate((phi, phi_bc[3][...,:,None]), axis=-1)
    if trim:
        return [phi_l[..., :-1,:], 
                phi_r[..., 1: ,:], 
                phi_b[..., :,:-1], 
                phi_t[..., :,1: ]]
    else:
        return [phi_l, phi_r, phi_b, phi_t]

def face_value_fu(data:dict, 
                  vkey:str,
                  scheme:str = 'upwind') -> Array:
    
    vel = jnp.stack([data['vx'],data['vy']], axis=1)
    vel_bc = [vel[:,:, 0,:], vel[:,:,-1,:],
              vel[:,:,:, 0], vel[:,:,:,-1]]                         ############################# WRONG: PERIODIC BC ########################################

    phi_cf   = stack_bc(data[vkey], data[vkey+'_bc'], trim=True)
    vel_cf   = stack_bc(vel, vel_bc, trim=True)
    if scheme == 'central':
        phi_face = [0.5* (data[vkey] + phi_i) for phi_i in phi_cf]
    elif scheme == 'upwind':
        # phi_face = [phi_cf[0], data[vkey], phi_cf[2], data[vkey]]
        phi_face = [0.5*(1+jnp.sign(vel_cf[0][:,0])) *phi_cf[0] + 0.5*(1-jnp.sign(vel_cf[0][:,0])) *data[vkey],
                    0.5*(1-jnp.sign(vel_cf[1][:,0])) *phi_cf[1] + 0.5*(1+jnp.sign(vel_cf[1][:,0])) *data[vkey],
                    0.5*(1+jnp.sign(vel_cf[2][:,1])) *phi_cf[2] + 0.5*(1-jnp.sign(vel_cf[2][:,1])) *data[vkey],
                    0.5*(1-jnp.sign(vel_cf[3][:,1])) *phi_cf[3] + 0.5*(1+jnp.sign(vel_cf[3][:,1])) *data[vkey]]
            
    phi_face = jnp.stack(phi_face, axis=1) 
    return phi_face

# def node_value_fu(data:dict) -> Array:
#     phi_cf     = stack_bc(data['phi'], data['phi_bc'])
#     phi_face = [0.5* (data['phi'] + phi_i) for phi_i in phi_cf]
#     phi_face = jnp.stack(phi_face, axis=1) 
#     return phi_face


def grad_phi_face_fu(data:dict,
                     vkey:str) -> Array:
    phi_cf     = stack_bc(data[vkey], data[vkey+'_bc'], trim=False)
    dx = jnp.mean(data['cell_x'][0,1: ,:]-data['cell_x'][0,:-1,:]) ############ WRONG #################################
    dy = jnp.mean(data['cell_x'][1,:,1: ]-data['cell_x'][1,:,:-1]) ############ WRONG #################################
    
    dpdx_fx = (phi_cf[1]-phi_cf[0])/dx
    dpdy_fy = (phi_cf[3]-phi_cf[2])/dy
    dpdX_fx = jnp.stack([dpdx_fx, jnp.zeros_like(dpdx_fx)],axis=1) ############ WRONG #################################
    dpdX_fy = jnp.stack([jnp.zeros_like(dpdy_fy), dpdy_fy],axis=1) ############ WRONG #################################
    dpdX_f  = jnp.stack([dpdX_fx[:,:,:-1,:], dpdX_fx[:,:,1:,:], 
                         dpdX_fy[:,:,:,:-1], dpdX_fy[:,:,:,1:]], axis=1)
    return dpdX_f


# def face_value_fu(scheme:str = 'upwind',
#                   **vargs) -> Array:
    
#     phi_cf   = stack_bc(vargs['phi'], vargs['phi_bc'], trim=True)
#     vel_cf   = stack_bc(vargs['adv'], vargs['vel_bc'], trim=True)
#     if scheme == 'central':
#         phi_face = [0.5* (vargs['phi'] + phi_i) for phi_i in phi_cf]
#     elif scheme == 'upwind':
#         # phi_face = [phi_cf[0], data['phi'], phi_cf[2], data['phi']]
#         phi_face = [0.5*(1+jnp.sign(vel_cf[0][:,0])) *phi_cf[0] + 0.5*(1-jnp.sign(vel_cf[0][:,0])) *vargs['phi'],
#                     0.5*(1-jnp.sign(vel_cf[1][:,0])) *phi_cf[1] + 0.5*(1+jnp.sign(vel_cf[1][:,0])) *vargs['phi'],
#                     0.5*(1+jnp.sign(vel_cf[2][:,1])) *phi_cf[2] + 0.5*(1-jnp.sign(vel_cf[2][:,1])) *vargs['phi'],
#                     0.5*(1-jnp.sign(vel_cf[3][:,1])) *phi_cf[3] + 0.5*(1+jnp.sign(vel_cf[3][:,1])) *vargs['phi']]
            
#     phi_face = jnp.stack(phi_face, axis=1) 
#     return phi_face

# def grad_phi_face_fu(data:dict,
#                      **vargs) -> Array:
#     phi_cf     = stack_bc(vargs['phi'], vargs['phi_bc'], trim=False)
#     dx = jnp.mean(data['cell_x'][0,1: ,:]-data['cell_x'][0,:-1,:]) ############ WRONG #################################
#     dy = jnp.mean(data['cell_x'][1,:,1: ]-data['cell_x'][1,:,:-1]) ############ WRONG #################################
    
#     dpdx_fx = (phi_cf[1]-phi_cf[0])/dx
#     dpdy_fy = (phi_cf[3]-phi_cf[2])/dy
#     dpdX_fx = jnp.stack([dpdx_fx, jnp.zeros_like(dpdx_fx)],axis=1) ############ WRONG #################################
#     dpdX_fy = jnp.stack([jnp.zeros_like(dpdy_fy), dpdy_fy],axis=1) ############ WRONG #################################
#     dpdX_f  = jnp.stack([dpdX_fx[:,:,:-1,:], dpdX_fx[:,:,1:,:], 
#                          dpdX_fy[:,:,:,:-1], dpdX_fy[:,:,:,1:]], axis=1)
#     return dpdX_f
