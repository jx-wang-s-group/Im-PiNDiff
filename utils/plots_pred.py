import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# from jax_cfd_stochastic_burgers_data_gen import val_data

# cmap = 'RdGy'
cmap = 'inferno'
clr = ['r','g','b','k','c','m','y','o']

# def plot_pred(args, data, datat_label, vkeys, case = 0):
#     name = ''
    
#     plt.close('all')
#     ord = (args['dims'][0][1]-args['dims'][0][0]) / (args['dims'][1][1]-args['dims'][1][0])
#     fig, ax = plt.subplots(3,2, figsize=(ord*len(vkeys)*5, 3*4))
    
#     for i, vkey in enumerate(vkeys):
#         levels = np.linspace(np.min(datat_label[vkey][case]), np.max(datat_label[vkey][case]+1e-3),20)
#         pcmu = ax[0,i].contourf(datat_label['cell_x'][0], datat_label['cell_x'][1], datat_label[vkey][case],  levels = levels)#, cmap=cmap)
#         fig.colorbar(pcmu , ax=ax[0,i], location='top')
#         pcmu = ax[1,i].contourf(data['cell_x'][0], data['cell_x'][1], data[vkey][case],  levels = levels)#, cmap=cmap)
#         # fig.colorbar(pcmu , ax=ax[1,i], location='right')
#         pcmu = ax[2,i].contourf(data['cell_x'][0], data['cell_x'][1], np.abs(datat_label[vkey][case]-data[vkey][case]))#, cmap=cmap)
#         # pcmu = ax[2].contourf(data['cell_x'][0], data['cell_x'][1], np.abs(datat_label[vkey][case]-data[vkey][case] / datat_label[vkey][case]))#, cmap=cmap)
#         fig.colorbar(pcmu , ax=ax[2,i], location='bottom')
#         name += vkey

#     # plt.savefig(args['path']+'/plots/data_rd')
#     plt.savefig(args['path']+'/plots/pred_'+name+'.png')


def plot_pred(args, data, datat_label, vkeys, nsample_times=5, case = 0):
    name = ''
    
    sample_times = [int(el) for el in np.linspace(0, len(data['datat'][vkeys[0]])-2, nsample_times)]

    plt.close('all')
    ord = (args['dims'][0][1]-args['dims'][0][0]) / (args['dims'][1][1]-args['dims'][1][0])
    fig, ax = plt.subplots(len(sample_times),3*len(vkeys), figsize=(ord*3*len(vkeys)*5, len(sample_times)*4), gridspec_kw = {'wspace':0, 'hspace':0})
    for k, vkey in enumerate(vkeys):
        # levels = np.linspace(np.min(datat_label[vkey][:,case]-1e-3), np.max(datat_label[vkey][:,case]+1e-3),20)
        for i,t in enumerate(sample_times):
            levels = np.linspace(np.min(datat_label['datat'][vkey][t,case]-1e-3), np.max(datat_label['datat'][vkey][t,case]+1e-3),20)
            im0 = ax[i,3*k+0].contourf(datat_label['cell_x'][0], datat_label['cell_x'][1], datat_label['datat'][vkey][t,case],  levels = levels, extend='both', cmap=cmap)
            im1 = ax[i,3*k+1].contourf(data['cell_x'][0], data['cell_x'][1], data['datat'][vkey][t,case],  levels = levels, extend='both', cmap=cmap)
            im2 = ax[i,3*k+2].contourf(data['cell_x'][0], data['cell_x'][1], 100*np.abs(datat_label['datat'][vkey][t,case]-data['datat'][vkey][t,case]) / np.mean(np.abs(datat_label['datat'][vkey][t,case])), cmap='Blues')
            # fig.colorbar(pcmu1 , ax=ax[i,3*k+2], location='left')
            # fig.colorbar(pcmu3 , ax=ax[i,3*k+2], location='right')
            for j in range(3):
                ax[i,3*k+j].axes.yaxis.set_ticklabels([])
                ax[i,3*k+j].axes.xaxis.set_ticklabels([])
                ax[i,3*k+j].tick_params(left = False)
                ax[i,3*k+j].tick_params(bottom=False)
        # fig.colorbar(pcmu1 , ax=ax[i,3*k+2], location='bottom')
        # fig.colorbar(pcmu3 , ax=ax[i,3*k+2], location='bottom')
        # fig.colorbar(pcmu3, ax=ax.ravel().tolist())
        name += '_'+vkey

    fig.subplots_adjust(right=0.85)
    phi_ax = fig.add_axes([0.86, 0.5, 0.01, 0.35])
    err_ax = fig.add_axes([0.86, 0.15, 0.01, 0.3])
    cbarp = fig.colorbar(im1, cax=phi_ax)
    cbare = fig.colorbar(im2, cax=err_ax)
    # cbarp.set_label(r'$\phi, \hat\phi$', fontsize=16)
    # cbare.set_label(r'$(\phi-\hat\phi)/\hat\phi$', fontsize=16)
    cbarp.ax.tick_params(labelsize=18)
    cbare.ax.tick_params(labelsize=18)
    cbarp.formatter = FormatStrFormatter('%.2f')
    cbare.formatter = FormatStrFormatter('%.2f')

    # plt.tight_layout()
    # plt.savefig(args['path']+'/plots/data_rd')
    plt.savefig(args['path']+'/plots/pred'+name+'.png')


def plot_pred_gif(args, data, datat_label, vkey, case = 0):
    name = ''
    
    sample_times =len(data['datat'][vkey])

    fig, ax = plt.subplots(figsize=(10,5))
    ims = []
    for i, t in enumerate(range(sample_times)):
        levels = np.linspace(np.min(datat_label['datat'][vkey][t,case]-1e-3), np.max(datat_label['datat'][vkey][t,case]+1e-3),20)
        # im = ax.contourf(datat_label['cell_x'][0], datat_label['cell_x'][1], datat_label['datat'][vkey][t,case],  levels = levels, extend='both', animated=True)#, cmap=cmap)
        im = ax.contourf(data['cell_x'][0], data['cell_x'][1], data['datat'][vkey][t,case],  levels = levels, extend='both', animated=True)#, cmap=cmap)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=200)
    writergif = animation.PillowWriter(fps=5)
    ani.save(f'sample_dis_{1}.gif',writer=writergif, dpi=300)

    # plt.close('all')
    # ord = (args['dims'][0][1]-args['dims'][0][0]) / (args['dims'][1][1]-args['dims'][1][0])
    # fig, ax = plt.subplots(len(sample_times),3*len(vkeys), figsize=(ord*3*len(vkeys)*5, len(sample_times)*4), gridspec_kw = {'wspace':0, 'hspace':0})
    # for k, vkey in enumerate(vkeys):
    #     # levels = np.linspace(np.min(datat_label[vkey][:,case]-1e-3), np.max(datat_label[vkey][:,case]+1e-3),20)
    #     for i,t in enumerate(sample_times):
    #         levels = np.linspace(np.min(datat_label['datat'][vkey][t,case]-1e-3), np.max(datat_label['datat'][vkey][t,case]+1e-3),20)
    #         pcmu1 = ax[i,3*k+0].contourf(datat_label['cell_x'][0], datat_label['cell_x'][1], datat_label['datat'][vkey][t,case],  levels = levels, extend='both')#, cmap=cmap)
    #         pcmu2 = ax[i,3*k+1].contourf(data['cell_x'][0], data['cell_x'][1], data['datat'][vkey][t,case],  levels = levels, extend='both')#, cmap=cmap)
    #         pcmu3 = ax[i,3*k+2].contourf(data['cell_x'][0], data['cell_x'][1], np.abs(datat_label['datat'][vkey][t,case]-data['datat'][vkey][t,case]), cmap=cmap)
    #         fig.colorbar(pcmu1 , ax=ax[i,3*k+2], location='left')
    #         fig.colorbar(pcmu3 , ax=ax[i,3*k+2], location='right')
    #         for j in range(3):
    #             ax[i,3*k+j].axes.yaxis.set_ticklabels([])
    #             ax[i,3*k+j].tick_params(left = False)
    #     name += '_'+vkey

    # plt.savefig(args['path']+'/plots/data_rd')
    # plt.savefig(args['path']+'/plots/pred'+name+'_gif.png')

def plot_1D_wtime(args, data, datat_label, vkey, nsample_times, case = 0):

    # xval, yval = val_data()

    sample_times = [int(el) for el in np.linspace(0, len(data['datat'][vkey])-1, nsample_times)]

    plt.close('all')
    fig, ax = plt.subplots(len(sample_times),1, figsize=(4*5, len(sample_times)*4), gridspec_kw = {'wspace':0, 'hspace':0})
    for i,t in enumerate(sample_times):
        # ax[i].plot(datat['cell_x'][0][:,datat[vkey].shape[-1]//2], datat[vkey][t,case,:,datat[vkey].shape[-1]//2])
        ax[i].plot(data['cell_x'][0][:,datat_label['datat'][vkey].shape[-1]//2], datat_label['datat'][vkey][t,case,:,datat_label['datat'][vkey].shape[-1]//2], '--'+clr[i])
        ax[i].plot(data['cell_x'][0][:,data['datat'][vkey].shape[-1]//2], data['datat'][vkey][t,case,:,data['datat'][vkey].shape[-1]//2], clr[i], label='time={:.2f}'.format(data['datat']['tarr'][t]))
        # plt.plot(xval, yval[i], '--'+clr[i], linewidth=3)

    plt.legend()
    plt.savefig(args['path']+'/plots/data1D_'+vkey+'.png')
