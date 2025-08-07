import jax
import jax.numpy as jnp
from functools import partial
from solver.boundary import update_BC

from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
   
u,l = 0.8, 0.2
# y_fu = lambda x: 0.5*(jnp.sin(0.5*jnp.pi*x) + 1.)*(u-l) + l
# y_grad = lambda x: 0.25*jnp.pi*jnp.cos(0.5*jnp.pi*x)*(u-l)
y_fu = lambda x: 0.5*(jnp.sin(jnp.pi*(x+1.)) + 1.)*(u-l) + l
y_grad = lambda x: 0.25*jnp.pi*jnp.cos(jnp.pi*(x+1.))*(u-l)
    
def initialze_fu(args:dict) -> dict:
    
    def adv_field_x(data, dynamic=2):
        tt = data['tcur'] * dynamic
        nwaves = 30
        subkeys = jax.random.split(jax.random.PRNGKey(10), 4)
        k   = jax.random.randint(subkeys[0], (2,nwaves), minval=0, maxval=4)
        phi = jax.random.uniform(subkeys[1], (2,nwaves), minval=0, maxval=4)
        omg = jax.random.uniform(subkeys[2], (2,nwaves), minval=-1, maxval=1)
        amp = jax.random.uniform(subkeys[3], (nwaves,), minval=-2, maxval=2)
        data_patn = 0
        # for kxi, kyi, pxi, pyi, oxi, oyi, ampi in zip(k[0], k[1], phi[0], phi[1], omg[0], omg[1], amp):
        #     data_patn += ampi*jnp.sin( (2*jnp.pi*(kxi+oxi*tt)*data['cell_x'][0]+pxi)
        #                         + (2*jnp.pi*(kyi-oyi*tt)*data['cell_x'][1]+pyi) + tt) 
        for kxi, kyi, pxi, pyi, oxi, oyi, ampi in zip(k[0], k[1], phi[0], phi[1], omg[0], omg[1], amp):
            data_patn += ampi*jnp.sin(  (2*jnp.pi*(kxi*data['cell_x'][0] + oxi*tt) + pxi)
                                +       (2*jnp.pi*(kyi*data['cell_x'][1] + oyi*tt) + pyi)) 
            
        data_patn  = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
        adv_fld    = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0)
        return adv_fld
    
    def adv_field_y(data, dynamic=2):
        tt = data['tcur'] * dynamic
        nwaves = 15
        subkeys = jax.random.split(jax.random.PRNGKey(20), 4)
        k   = jax.random.randint(subkeys[0], (2,nwaves), minval=0, maxval=2)
        phi = jax.random.uniform(subkeys[1], (2,nwaves), minval=0, maxval=2)
        omg = jax.random.uniform(subkeys[2], (2,nwaves), minval=-1, maxval=1)
        amp = jax.random.uniform(subkeys[3], (nwaves,), minval=-2, maxval=2)
        data_patn = 0
        # for kxi, kyi, pxi, pyi, oxi, oyi, ampi in zip(k[0], k[1], phi[0], phi[1], omg[0], omg[1], amp):
        #     data_patn += ampi*jnp.sin( (2*jnp.pi*(kxi+oxi*tt)*data['cell_x'][0]+pxi)
        #                         + (2*jnp.pi*(kyi-oyi*tt)*data['cell_x'][1]+pyi) + tt) 
        for kxi, kyi, pxi, pyi, oxi, oyi, ampi in zip(k[0], k[1], phi[0], phi[1], omg[0], omg[1], amp):
            data_patn += ampi*jnp.sin(  (2*jnp.pi*(kxi*data['cell_x'][0] + oxi*tt) + pxi)
                                +       (2*jnp.pi*(kyi*data['cell_x'][1] + oyi*tt) + pyi)) 
            
        data_patn  = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
        adv_fld    = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0)
        return adv_fld
    
    def diff_field_x(data, dynamic=2):
        tt = data['tcur'] * dynamic
        nwaves = 20
        subkeys = jax.random.split(jax.random.PRNGKey(10), 4)
        k   = jax.random.randint(subkeys[0], (2,nwaves), minval=0, maxval=4)
        phi = jax.random.uniform(subkeys[1], (2,nwaves), minval=0, maxval=4)
        omg = jax.random.uniform(subkeys[2], (2,nwaves), minval=-1, maxval=1)
        amp = jax.random.uniform(subkeys[3], (nwaves,), minval=-2, maxval=2)
        data_patn = 0
        # for kxi, kyi, pxi, pyi, oxi, oyi, ampi in zip(k[0], k[1], phi[0], phi[1], omg[0], omg[1], amp):
        #     data_patn += ampi*jnp.sin( (2*jnp.pi*(kxi+oxi*tt)*data['cell_x'][0]+pxi)
        #                         + (2*jnp.pi*(kyi-oyi*tt)*data['cell_x'][1]+pyi) + tt)
        for kxi, kyi, pxi, pyi, oxi, oyi, ampi in zip(k[0], k[1], phi[0], phi[1], omg[0], omg[1], amp):
            data_patn += ampi*jnp.sin(  (2*jnp.pi*(kxi*data['cell_x'][0] + oxi*tt) + pxi)
                                +       (2*jnp.pi*(kyi*data['cell_x'][1] + oyi*tt) + pyi))  
            
        data_patn  = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
        diff_fld   = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0)
        return diff_fld
    
    # def diff_field_y(data, dynamic=2):
    #     tt = data['tcur'] * dynamic
    #     nwaves = 15
    #     subkeys = jax.random.split(jax.random.PRNGKey(20), 4)
    #     k   = jax.random.randint(subkeys[0], (2,nwaves), minval=0, maxval=2)
    #     phi = jax.random.uniform(subkeys[1], (2,nwaves), minval=0, maxval=2)
    #     omg = jax.random.uniform(subkeys[2], (2,nwaves), minval=-1, maxval=1)
    #     amp = jax.random.uniform(subkeys[3], (nwaves,), minval=-2, maxval=2)
    #     data_patn = 0
    #     # for kxi, kyi, pxi, pyi, oxi, oyi, ampi in zip(k[0], k[1], phi[0], phi[1], omg[0], omg[1], amp):
    #     #     data_patn += ampi*jnp.sin( (2*jnp.pi*(kxi+oxi*tt)*data['cell_x'][0]+pxi)
    #     #                         + (2*jnp.pi*(kyi-oyi*tt)*data['cell_x'][1]+pyi) + tt) 
    #     for kxi, kyi, pxi, pyi, oxi, oyi, ampi in zip(k[0], k[1], phi[0], phi[1], omg[0], omg[1], amp):
    #         data_patn += ampi*jnp.sin(  (2*jnp.pi*(kxi*data['cell_x'][0] + oxi*tt) + pxi)
    #                             +       (2*jnp.pi*(kyi*data['cell_x'][1] + oyi*tt) + pyi)) 
            
    #     data_patn  = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
    #     diff_fld   = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0)
    #     return diff_fld
    
    # def adv_field_x(data, dynamic=-10):
    #     # pipe flow
    #     # d = (y_fu(data['cell_x'][0]) - data['cell_x'][1])**2
    #     # dydx = y_grad(data['cell_x'][0])
    #     # data_patn  = jax.nn.relu(0.03-d) * jnp.cos(jnp.arctan(dydx))

    #     gauss = 1#jnp.exp(-((data['cell_x'][0]-0.)**2+4*(data['cell_x'][1]-0.5)**2)/0.5)
    #     tt = data['tcur'] * dynamic
    #     data_patn  = (2+jnp.sin(20*(data['cell_x'][0]+tt)**2 + 30*(data['cell_x'][1]-0.5+0.5*tt)**2)) * gauss

    #     data_patn  = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
    #     adv_fld    = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0)
    #     return adv_fld# * 100
         
    # def adv_field_y(data, dynamic=0):
    #     # pipe flow
    #     # d = (y_fu(data['cell_x'][0]) - data['cell_x'][1])**2
    #     # dydx = y_grad(data['cell_x'][0])
    #     # data_patn  = jax.nn.relu(0.03-d) * jnp.sin(jnp.arctan(dydx))

    #     # gauss = 1#jnp.exp(-((data['cell_x'][0]-0.)**2+4*(data['cell_x'][1]-0.5)**2)/0.5)
    #     # tt = data['tcur'] * dynamic
    #     # data_patn  = (2+jnp.sin(20*(data['cell_x'][0]-tt)**2 + 30*(data['cell_x'][1]-0.5-0.5*tt)**2)) * gauss
    #     data_patn  = 30*jnp.cos(100*(data['cell_x'][1]-0.5)**2)

    #     data_patn  = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
    #     adv_fld    = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0)
    #     return adv_fld# * 100
    
    # def diff_field_x(data):
    #     gauss = 1#jnp.exp(-((data['cell_x'][0]-0.)**2+4*(data['cell_x'][1]-0.5)**2)/0.5)
    #     tt = 10*data['tcur']
    #     data_patn  = (2+jnp.sin(20*(data['cell_x'][0]+tt)**2 + 30*(data['cell_x'][1]-0.5+0.5*tt)**2)) * gauss
    #     data_patn  = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
    #     diff_fld   = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0)
    #     return diff_fld#*1e-1
         
    # def diff_field_y(data):
    #     gauss = 1#jnp.exp(-((data['cell_x'][0]-0.)**2+4*(data['cell_x'][1]-0.5)**2)/0.5)
    #     tt = 00*data['tcur']
    #     k = jax.random.randint(jax.random.PRNGKey(10), (2,30), minval=0, maxval=5)
    #     phi = jax.random.uniform(jax.random.PRNGKey(11), (2,30), minval=0, maxval=5)
    #     data_patn = 0
    #     for kxi, kyi, pxi, pyi in zip(k[0], k[1], phi[0], phi[1]):
    #         data_patn += jnp.sin( (2*jnp.pi*(kxi+0.5*tt)*data['cell_x'][0]+pxi)
    #                             + (2*jnp.pi*(kyi-0.5*tt)*data['cell_x'][1]+pyi) + tt) 
            
    #     # data_patn  = (2+jnp.sin(k[0]*data['cell_x'][0]-k[1]*data['cell_x'][1]-0.5*tt))
    #     # data_patn  = (2+jnp.sin(20*(data['cell_x'][0]-tt)**2 + 30*(data['cell_x'][1]-0.5-0.5*tt)**2)) * gauss

    #     data_patn  = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
    #     diff_fld   = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0)
    #     return diff_fld#*1e-1
    
    # advection fields functions
    if args['adv'][0] == 'call_fu':
        args['vel_fu'][0] = partial(adv_field_x, dynamic=0 if 'steady' in args['case_setup'] else 1)
    else:
        args['vel_fu'][0] = lambda data: args['adv'][0] * jnp.ones([args['nBatch'],*args['nCell']])    
    
    if args['adv'][1] == 'call_fu':
        args['vel_fu'][1] = partial(adv_field_y, dynamic=0 if 'steady' in args['case_setup'] else 1)
    else:
        args['vel_fu'][1] = lambda data: args['adv'][1] * jnp.ones([args['nBatch'],*args['nCell']])    

    # Diffusion fields functions
    if args['kdd'][0] == 'call_fu':
        args['diff_fu'][0] = partial(diff_field_x, dynamic=0 if 'steady' in args['case_setup'] else 0.1)
    else:
        args['diff_fu'][0] = lambda data: args['kdd'][0] * jnp.ones([args['nBatch'],*args['nCell']])    
    
    if args['kdd'][1] == 'call_fu':
        args['diff_fu'][1] = partial(diff_field_x, dynamic=0 if 'steady' in args['case_setup'] else 0.1)
    else:
        args['diff_fu'][1] = lambda data: args['kdd'][1] * jnp.ones([args['nBatch'],*args['nCell']])    

    return args

def initialze_data(args:dict,
                   data:dict) -> dict:

    data['tcur'] = jnp.array(0)
    data['dt']   = args['dt']
    data['Tend'] = jnp.array(args['Tend'])

    # ## Gaussian Process
    # gp_vel = []
    # for i in range(2):
    #     gp_vel_i = gaussian_distribution(jax.random.PRNGKey(100*i), args, data, n1 = [40,20], ic='GP', length_scale_bounds=(0.5, 10.01))
    #     gp_vel_i = (gp_vel_i - jnp.mean(gp_vel_i)) / (3*jnp.std(gp_vel_i))
    #     gp_vel.append(gp_vel_i)
    # # args['gp_vel'] = jnp.array(gp_vel)#+0.1
    # args['gp_vel'] = (jnp.array(gp_vel)>0)*1.9 + 1.1 
    
    # data_ic = jnp.repeat(((jnp.abs(data['cell_x'][0])<0.2)*1.0)[None,:],  args['nBatch'], axis=0) 

    # data_ic = jnp.exp(-((data['cell_x'][0]-0)**2+(data['cell_x'][1]-0.5)**2) / (args['dims'][0][1]/50))
    # data_ic = jnp.exp(- (data['cell_x'][0]-0.5*args['dims'][0][1])**2 / (args['dims'][0][1]/50))
    # data_ic = jnp.exp(-(data['cell_x'][0]-0.5*0)**2 / (2/50))
    # data_ic = jnp.repeat(data_ic[None], args['nBatch'], axis=0)

    ## Gaussian Process
    # Training
    data_ic = []
    for i in range(args['nBatch']//2):
        frac = 2*i/args['nBatch']
        # data_ic_i = gaussian_distribution(jax.random.PRNGKey(10*i), args, data, n1 = [30,10], ic='GP', length_scale_bounds=(0.1+1.*frac, 10.01))
        data_ic_i = gaussian_distribution(jax.random.PRNGKey(10*i), args, data, n1 = [30,10], ic='GP', length_scale_bounds=(0.3+0.1*frac, 10.01))
        data_ic_i = (data_ic_i - jnp.min(data_ic_i)) / (jnp.max(data_ic_i) - jnp.min(data_ic_i))
        data_ic.append(data_ic_i)
    # Testing
    for i in range(args['nBatch']//2):
        frac = 2*i/args['nBatch']
        data_ic_i = gaussian_distribution(jax.random.PRNGKey(10*i), args, data, n1 = [30,10], ic='GP', length_scale_bounds=(0.1+1.*frac, 10.01))
        data_ic_i = (data_ic_i - jnp.min(data_ic_i)) / (jnp.max(data_ic_i) - jnp.min(data_ic_i))
        data_ic.append(data_ic_i)
    data_ic = jnp.array(data_ic)
    # data_ic = jnp.clip(data_ic, a_min=0.5)

    ## pipe flow
    # d = (y_fu(data['cell_x'][0]) - data['cell_x'][1])**2
    # data_ic  = 100*jax.nn.relu(0.03-d) * data_ic

    data_patn   = (2+jnp.sin(10*data['cell_x'][0])+jnp.cos(15*data['cell_x'][1]))
    # data_patn = gaussian_distribution(jax.random.PRNGKey(1000), args, data, n1 = [30,10], ic='GP', length_scale_bounds=(0.5, 10.01))
    data_patnx = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 
    data_patn   = (2+jnp.sin(10*data['cell_x'][0]+15*data['cell_x'][1]))
    # data_patn = gaussian_distribution(jax.random.PRNGKey(2000), args, data, n1 = [30,10], ic='GP', length_scale_bounds=(0.5, 10.01))
    data_patny = (data_patn - jnp.min(data_patn)) / (jnp.max(data_patn) - jnp.min(data_patn)) 



    if "AdvDiff" in args['case_setup']:
        
        # # advection fields functions
        # if args['adv'][0] == 'call_fu':
        #     args['vel_fu'][0] = adv_field_x
        # else:
        #     args['vel_fu'][0] = lambda data: args['adv'][0] * jnp.ones([args['nBatch'],*args['nCell']])    
        
        # if args['adv'][1] == 'call_fu':
        #     args['vel_fu'][1] = adv_field_y
        # else:
        #     args['vel_fu'][1] = lambda data: args['adv'][1] * jnp.ones([args['nBatch'],*args['nCell']])    

        data['phi'] = data_ic
        data['vx']  = args['vel_fu'][0](data) 
        data['vy']  = args['vel_fu'][1](data) 

    elif "Burgers1v" in args['case_setup']:
        data['vx']          = data_ic
        data['vx_pre']      = data_ic
        data['vx_pre_bc']   = [0]*4
        data['vy']          = jnp.zeros_like(data_ic)
        data['vy_pre']      = jnp.zeros_like(data_ic)
        data['vy_pre_bc']   = [0]*4
        data = update_BC(data, ['vx_pre','vy_pre'])

    elif "Burgers" in args['case_setup']:
        args['state_var'] = ['vx','vy']
        data['vx']  = data_ic
        data['vy']  = jnp.zeros_like(data_ic)

    for vkey in ['vx','vy',*args['state_var']]:
        data[vkey+'_bc'] = [0]*4
    data = update_BC(data, vkeys=['vx','vy',*args['state_var']])


    # # Diffusion fields functions
    # if args['kdd'][0] == 'call_fu':
    #     args['diff_fu'][0] = diff_field_x
    # else:
    #     args['diff_fu'][0] = lambda data: args['kdd'][0] * jnp.ones([args['nBatch'],*args['nCell']])    
    
    # if args['kdd'][1] == 'call_fu':
    #     args['diff_fu'][1] = diff_field_y
    # else:
    #     args['diff_fu'][1] = lambda data: args['kdd'][1] * jnp.ones([args['nBatch'],*args['nCell']])    

    data['kx']  = args['diff_fu'][0](data)
    data['ky']  = args['diff_fu'][1](data)
    # data['kx']  = jnp.ones([args['nBatch'],*args['nCell']]) *args['kdd'][0]
    # data['ky']  = jnp.ones([args['nBatch'],*args['nCell']]) *args['kdd'][1]
    # data['kx']  = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0) *args['kdd'][0]
    # data['ky']  = jnp.repeat(data_patn[None,:],  args['nBatch'], axis=0) *args['kdd'][1]
    data['kx_bc'], data['ky_bc'] = [0]*4, [0]*4
    data = update_BC(data, ['kx','ky'], bc_type = 'gradFree')
    
    return data


def gaussian_distribution(key, args, data, n1, ic='GP', length_scale_bounds=(1., 10.01)):
    """
    Function to initial fields on grid 
    """
    if 'random' in ic:
        skey1, skey2 = jax.random.split(key)
        phi = jax.random.uniform(subkey, shape=data['nCell'])
    elif 'GP' in ic:
        key, subkey = jax.random.split(key)
        n2 = args['nCell']  # Number of points in posterior (test points)
        ny = 1  # Number of functions that will be sampled from the posterior
        domain = args['dims']

        # Sample observations (X1, y1) on the function
        X11 = jnp.linspace(domain[0][0], domain[0][1], n1[0])
        X12 = jnp.linspace(domain[1][0], domain[1][1], n1[1])
        X1  = jnp.stack(jnp.meshgrid(X11, X12, indexing='xy')).reshape(2,(n1[0]*n1[1])).T
        y1  = jax.random.uniform(subkey, shape=(n1[0]*n1[1],))
        X2  = data['cell_x'].reshape(2,n2[0]*n2[1]).T


        kern = RBF(length_scale=1., length_scale_bounds=length_scale_bounds)
        gpc1 = GaussianProcessRegressor(kernel=kern, random_state=0).fit(X1, y1)
        y2_mean2 = gpc1.predict(X2, return_std=False, return_cov=False)
        # y2 = gpc1.sample_y(X2, n_samples=ny, random_state=0)[:,0]
    return y2_mean2.reshape(n2[0],n2[1])


