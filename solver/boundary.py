
def update_BC(data:dict,
              vkeys:list,
              bc_type:str = 'periodic') -> dict:

    if bc_type == 'periodic':
        for vkey in vkeys:
            data[vkey+'_bc'][0] = data[vkey][:,-1,:]
            data[vkey+'_bc'][1] = data[vkey][:, 0,:]
            data[vkey+'_bc'][2] = + data[vkey][:,:,-1]
            data[vkey+'_bc'][3] = + data[vkey][:,:, 0]
            # data[vkey+'_bc'][2] = + data[vkey][:,:, 0]
            # data[vkey+'_bc'][3] = + data[vkey][:,:,-1]

    if bc_type == 'gradFree':
        for vkey in vkeys:
            data[vkey+'_bc'][0] = data[vkey][:, 0,:]
            data[vkey+'_bc'][1] = data[vkey][:,-1,:]
            data[vkey+'_bc'][2] = + data[vkey][:,:, 0]
            data[vkey+'_bc'][3] = + data[vkey][:,:,-1]
            # data[vkey+'_bc'][2] = + data[vkey][:,:, 0]
            # data[vkey+'_bc'][3] = + data[vkey][:,:,-1]

    return data
