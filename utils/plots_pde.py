import numpy as np
import matplotlib.pyplot as plt

# from jax_cfd_stochastic_burgers_data_gen import val_data

# cmap = 'RdGy'
cmap = 'Reds'
clr = ['r','g','b','k','c','m','y','o']

def plot_pde_wtime(args, data, vkeys, nsample_times, case = 0):

    sample_times = [int(el) for el in np.linspace(0, len(data['datat'][vkeys[0]])-1, nsample_times)]

    name = ''
    plt.close('all')
    ord = (args['dims'][0][1]-args['dims'][0][0]) / (args['dims'][1][1]-args['dims'][1][0])
    fig, ax = plt.subplots(len(sample_times),len(vkeys), figsize=(ord*len(vkeys)*5, len(sample_times)*4))
    for k, vkey in enumerate(vkeys):
        # Udmin, Udmax = np.min(data['datat'][vkey]), np.max(data['datat'][vkey])
        for i,t in enumerate(sample_times):
            pcmu = ax[i,k].contourf(data['cell_x'][0], data['cell_x'][1], data['datat'][vkey][t,case], 
                                levels = np.linspace(np.min(data['datat'][vkey][t,case]), np.max(data['datat'][vkey][t,case]+1e-3),20))#, cmap=cmap)
            fig.colorbar(pcmu , ax=ax[i,k], location='right')
        name += '_'+vkey

    # plt.savefig(args['path']+'/plots/data_rd')
    plt.savefig(args['path']+'/plots/data'+name+'.png')


def plot_1D_wtime(args, data, vkey, nsample_times, case = 0):

    # xval, yval = val_data()

    sample_times = [int(el) for el in np.linspace(0, len(data['datat'][vkey])-1, nsample_times)]

    plt.close('all')
    # fig, ax = plt.subplots(len(sample_times),1, figsize=(3*5, len(sample_times)*4))
    for i,t in enumerate(sample_times):
        # ax[i].plot(datat['cell_x'][0][:,datat[vkey].shape[-1]//2], datat[vkey][t,case,:,datat[vkey].shape[-1]//2])
        plt.plot(data['cell_x'][0][:,data['datat'][vkey].shape[-1]//2], data['datat'][vkey][t,case,:,data['datat'][vkey].shape[-1]//2], clr[i], label='time={:.2f}'.format(data['datat']['tarr'][t]))
        # plt.plot(xval, yval[i], '--'+clr[i], linewidth=3)

    plt.legend()
    plt.savefig(args['path']+'/plots/data1D_'+vkey+'.png')



# def plot_1D_wtime(args, data, vkey, nsample_times, case = 0):

#     # xval, yval = val_data()
#     fac = 100
#     cx = np.linspace(args['dims'][0][0], args['dims'][0][1], args['nCell'][0]*fac)
#     time_arr = np.linspace(0,args['Tend'], int(np.ceil(args['Tend']/args['dt']))+1)

#     sample_times = [int(el) for el in np.linspace(0, len(data['datat'][vkey])-1, nsample_times)]

#     plt.close('all')
#     fig, ax = plt.subplots(len(sample_times),1, figsize=(4*5, len(sample_times)*4), gridspec_kw = {'wspace':0, 'hspace':0})
#     for i,t in enumerate(sample_times):
#         true_phi = np.logical_and(((-0.2+args['adv'][0]*time_arr[t])<cx), (cx<(0.2+args['adv'][0]*time_arr[t])))*1.0
#         pred_phi = data['datat'][vkey][t,case,:,data['datat'][vkey].shape[-1]//2]
#         pred_phi = np.repeat(pred_phi[None], fac, axis=0)
#         pred_phi = pred_phi.T.reshape(-1)

#         # ax[i].plot(datat['cell_x'][0][:,datat[vkey].shape[-1]//2], datat[vkey][t,case,:,datat[vkey].shape[-1]//2])
#         ax[i].plot(cx, true_phi, '--'+clr[i])
#         ax[i].plot(cx, pred_phi, clr[i], label='time={:.2f}'.format(data['datat']['tarr'][t]))
#         ax[i].set_xticks(data['node_x'][0][:,data['datat'][vkey].shape[-1]//2], minor=True)
#         ax[i].xaxis.grid(True, which='minor')
#         # plt.plot(xval, yval[i], '--'+clr[i], linewidth=3)

#     plt.legend()
#     plt.savefig(args['path']+'/plots/data1D_'+vkey+'.png')