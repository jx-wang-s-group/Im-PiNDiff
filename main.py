# env = jax3
import os, errno, time, copy
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import optax

# from solver.rd_pde import *
from solver.mesh import grid
from solver.diff_eq_solver import get_roleout
from solver.initalizer import initialze_fu, initialze_data
from nn.trainer import *
from nn.NN_utils import get_bias_flat2paratree, get_flat2paratree
from nn.flax_nn import MLP_Net, MLP_ParaNet_LastLayer
from utils.utils import PyTree
from utils.plots_pde import plot_pde_wtime, plot_1D_wtime

from typing import Callable
from jax._src.typing import Array


# jax.config.update("jax_debug_nans", True)

HOME = os.getcwd()

def add_err(key: Array, 
            Data: dict):

    key, subkey = random.split(key)
    white_err =  5e-2*random.normal(subkey, shape=Data.shape)
    Data = Data.at[1:].set(Data[1:] + white_err[1:])
    return key, Data


def get_Data(key: Array, 
             args: dict, 
             params: dict,
             **kwargs) -> (Array, dict):
    
    data = {}
    data.update(grid(args))
    data_ICBC = initialze_data(args, data)
    roleout = get_roleout(args, {}, sim_tarr=args['sim_tarr'], debug=args['debug'])
    starttime = time.time()
    data, sol_info = roleout(params, data_ICBC)
    endtime = time.time()
    print("Data gen time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))

    if "AdvDiff" in args['case_setup']:
        plot_pde_wtime(args, data, vkeys=['phi', *args['train']],nsample_times=5, case = 0)
        plot_1D_wtime( args, data, vkey='phi',nsample_times=5, case = 0)
        # plot_pde_wtime(args, data, vkey='vx', nsample_times=5, case = -1)
        # plot_pde_wtime(args, data, vkey='vy', nsample_times=5, case = -1)
    elif "Burgers" in args['case_setup']:
        plot_pde_wtime(args, data, vkeys=['vx', 'kx', 'ky'], nsample_times=5, case = -1)
        plot_1D_wtime( args, data, vkey ='vx', nsample_times=5, case = 0)
        # plot_pde_wtime(data, vkey='vy', nsample_times=5, case = -1)
        # plot_1D_wtime( data, vkey='vy', nsample_times=5, case = -1)

    return key, data


def solve_pdesystem(key: Array, 
                    args: dict, 
                    datat_label: dict, 
                    **kwargs):

    key, subkey = random.split(key)

    NFmodel, NFparams = {}, {}
    PJmodel, PJparams = {}, {}
    for vkey in args['train']:
        key, subkey1, subkey2 = random.split(key, 3)
        NFmodel[vkey]  = MLP_Net(layers=args['NF_layers'], activation=['sin','relu'])
        NFparams[vkey] = NFmodel[vkey].get_NNparams(subkey1)
        # ## Film
        # _, tot_para = get_bias_flat2paratree(AdvNFparams_tree[vkey])
        ## Full projection
        _, tot_para    = get_flat2paratree(NFparams[vkey])
        PJmodel[vkey]  = MLP_ParaNet_LastLayer(layers=[args['HY_layers'][-1], tot_para], NF_layers=args['NF_layers'], NFparams_tree = NFparams[vkey])
        PJparams[vkey] = PJmodel[vkey].get_NNparams(subkey2)


    AdvNFparams_tree = copy.deepcopy(NFparams)
    # AdvNFparams_tree = jax.tree_util.tree_map(lambda x: jnp.ones_like(x)*(len(x.shape)==1), AdvNFparams_tree)
    args['NFparams_tree'] = AdvNFparams_tree
    


    HYmodel  = MLP_Net(layers=args['HY_layers'], activation=['leaky_relu',None])
    HYparams = HYmodel.get_NNparams(subkey)

    # y = AdvNFmodel(AdvNFparams, x)

    models = {'NF':(HYmodel,  PJmodel,  NFmodel)}
    params = {'NF':(HYparams, PJparams, NFparams)}#, 'condel': 1.e1}
    mask   = {'NF':1}

    lr, cd = args['learning_rate'], 1e-3
    # cosdecay  = optax.cosine_decay_schedule(lr, decay_steps=args['nepochs'], alpha=cd)
    nodecay   = optax.constant_schedule(lr)
    cosdecay  = optax.cosine_decay_schedule(lr, decay_steps=args['nepochs']//4*3, alpha=cd)
    jointopt  = optax.join_schedules([nodecay, cosdecay], boundaries=[args['nepochs']//4])
    opt_fu    = {'NF':optax.adam(learning_rate=jointopt)}#, 'condel':optax.adam(learning_rate=1.0)}
    opt_state = {'NF':opt_fu['NF'].init(params['NF'])}#, 'condel':opt_fu['condel'].init(params['condel'])}

    trainable = (models, opt_fu, params, mask, opt_state)

    starttime = time.time()
    params, loss_list = train(args, trainable,datat_label, **kwargs)
    endtime = time.time()
    print("Train time = ",time.strftime("%H:%M:%S", time.gmtime(endtime - starttime)))

    plt.close('all')
    plt.plot(jnp.log(jnp.array(loss_list)))
    plt.savefig(args['path']+'/plots/loss.png')
    


if __name__ == "__main__":

    key  = random.PRNGKey(123)

    prob_typ  = "AdvDiff"               # "AdvDiff" or "Burgers1v"
    stddynmc  = 'steady'                # "steady", "dynamic"
    gen_data  = False                   # True to generate data, False to train model
    debug     = False                   # True to run in debug mode (all jit and and scan are desibled, slower)

    args = dict(
        gen_data = gen_data,
        debug = debug,
        case_setup = prob_typ + '_' + stddynmc,
        
        nCell = [128,64],
        dims = [[-1,1], [0, 1]],
        
        nBatch = 10,
        NF_layers = [2, 128, 1],
        HY_layers = [1, 16,16,16, 10],
        epochstart = 0,
        vel_fu = [None]*2,
        diff_fu = [None]*2,
        ord = {'vx': 1., 'vy': 1., 'kx': 1e-2, 'ky': 1e-2},
        path = os.path.join(HOME, "output/"+prob_typ+'_'+stddynmc),
    )

    
    if args['gen_data']:
        args['odesolve']  = 'rk4'        # FwdEuler, BackEuler, CrankNicolson, rk4
        args['dt']        = 1e-3         # time-step

    else:
        args['odesolve']  = 'CrankNicolson'   # FwdEuler, BackEuler, CrankNicolson, rk4
        args['dt']        = 1e-2         # time-step
        args['plot_depoch']   = 100
        args['learning_rate'] = 1e-4

    if 'AdvDiff' in args['case_setup']:
        args['train'] = ['vx', 'vy']
        args['adv'] = ['call_fu', 'call_fu']
        args['kdd'] = [1e-2, 1e-2] 
        
        args['state_var'] = ['phi']
        args['trac_var']  = ['vx', 'vy', 'kx', 'ky', 'phi']

        if 'steady' in args['case_setup']:
            args['Tend']     = 0.2
            args['nepochs']  = 2000
            args['sim_tarr'] = jnp.arange(0,int(jnp.ceil(args['Tend']/args['dt']))+1)*args['dt']
            train_time       = [0.05]
        elif 'dynamic' in args['case_setup']:
            args['Tend']     = 0.4
            args['nepochs']  = 10_000
            args['sim_tarr'] = jnp.array([0.  , 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.102,
                                                0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.201,
                                                0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.298,
                                                0.31, 0.32, 0.33, 0.34, 0.35, 0.36, 0.37, 0.38, 0.39, 0.4])
            train_time       = [0.05, 0.102, 0.15, 0.201, 0.25, 0.298, 0.35]

    elif 'Burgers1v' in args['case_setup']:
        args['Tend']  = 1.
        args['train'] = ['kx']
        args['adv'] = [1., .8]
        args['kdd'] = ['call_fu', 'call_fu'] 
        
        args['state_var'] = ['vx']
        args['trac_var']  = ['vx', 'kx', 'ky']
    
        if 'steady' in args['case_setup']:
            args['nepochs']  = 2000
            train_time       = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
            args['sim_tarr'] = jnp.arange(0,int(jnp.ceil(args['Tend']/args['dt']))+1)*args['dt']
        elif 'dynamic' in args['case_setup']:
            args['nepochs']  = 10_000
            train_time       = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
            args['sim_tarr'] = jnp.arange(0,int(jnp.ceil(args['Tend']/args['dt']))+1)*args['dt']


    if args['debug']: print('***************** Running debug mode *****************')


    dir = args['case_setup']
    print('case = '+dir)
    args['path'] = os.path.join(HOME, "output/"+dir)
    try:
        os.makedirs(args['path'])
        os.makedirs(args['path']+'/plots')
        os.makedirs(args['path']+'/checkpoints')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    
    args = initialze_fu(args)
    


    if args['gen_data']:
        key, datat = get_Data(key, args, params={})
        PyTree.save(datat, args['path']+'/checkpoints', name='datat_label')
        del args['vel_fu'], args['diff_fu']
        PyTree.save(args,  args['path']+'/checkpoints', name='args')

    elif args['train']:

        fac = 1000
        datat_label = PyTree.load(args['path']+'/checkpoints', name='datat_label')
        data_tarr_list  = (jnp.rint(fac*datat_label['datat']['tarr'])).tolist()
        sim_tarr_list = jnp.rint( fac*args['sim_tarr'] ).tolist()
        sim_idx = []
        for sim_t in sim_tarr_list:
            sim_idx.append(data_tarr_list.index(sim_t))
        datat_label['datat'] = jax.tree_util.tree_map(lambda x: x[sim_idx,], datat_label['datat'])

        if train_time:
            data_time_list  = (jnp.rint(fac*datat_label['datat']['tarr'])).tolist()
            traintime  = []
            for train_t in train_time:
                didx = data_time_list.index(train_t*fac)
                sidx = sim_tarr_list.index(train_t*fac)
                assert(didx == sidx), "idx dont match"
                traintime.append(sidx)
        else:
            traintime = None

        trainbatch = jnp.arange(5)

        solve_pdesystem(key, args, datat_label, traintime=traintime, trainbatch=trainbatch)
