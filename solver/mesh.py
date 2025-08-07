import jax.numpy as jnp
import jax
import numpy as np

tol = 1e-6


def grid(args:dict) -> dict:

    ndim = len(args['nCell'])
    nNodes = [args['nCell'][i]+1 for i in range(len(args['nCell']))]

    node_x = [jnp.linspace(*args['dims'][i], nNodes[i]) for i in range(len(args['nCell']))]
    node_x = np.meshgrid(*node_x, indexing='ij')
    
    if ndim == 1:
        cell_x  = [0.5*(node_x[0][1:]+node_x[0][:-1])]
        cell_dx = [node_x[0][1:]-node_x[0][:-1]]

    elif ndim == 2:
        cell_x  = [0.25*(node_x[i][1: ,1: ]+
                         node_x[i][:-1,1: ]+
                         node_x[i][:-1,:-1]+
                         node_x[i][1: ,:-1]) for i in range(2)]
        cell_dx_e = [node_x[i][1: ,1: ]-node_x[i][1: ,:-1] for i in range(2)]
        cell_dx_n = [node_x[i][:-1,1: ]-node_x[i][1: ,1: ] for i in range(2)]
        cell_dx_w = [node_x[i][:-1,:-1]-node_x[i][:-1,1: ] for i in range(2)]
        cell_dx_s = [node_x[i][1:,:-1 ]-node_x[i][:-1,:-1] for i in range(2)]

        cell_dx = [cell_dx_w, cell_dx_e, cell_dx_s, cell_dx_n]
        face_A  = [[dx[1], - dx[0]] for dx in cell_dx]

        cell_dx = [[jnp.abs(dx[0]), jnp.abs(dx[1])] for dx in cell_dx]
        cell_vol = - cell_dx_s[0] * cell_dx_w[1] + cell_dx_s[1] * cell_dx_w[0]
        
    elif ndim == 3:
        cell_x  = [0.5*(node_x[0][1:,:,:]+node_x[0][:-1,:,:]),
                   0.5*(node_x[1][:,1:,:]+node_x[1][:,:-1,:]),
                   0.5*(node_x[2][:,:,1:]+node_x[2][:,:,:-1])]
        
        
    Grid = {'nCell':   args['nCell'],
            'nNodes':  nNodes,
            'node_x':  node_x,
            'cell_x':  cell_x, 
            'cell_dx': cell_dx,
            'face_A':  face_A, 
            'cell_vol':cell_vol}
    
    for k in Grid:
        if not isinstance(Grid[k], str):
            Grid[k] = jnp.array(Grid[k])

    return Grid




if __name__ == "__main__":

    args = {}
    args['nCell']    = [5,5]
    args['dims']      = [[0, 1], [-1, 1]]

    Grid = grid(args)
    print(Grid['cell_x'].shape)
