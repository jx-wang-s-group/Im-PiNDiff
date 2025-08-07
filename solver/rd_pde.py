import jax
import jax.numpy as jnp

from utils.utils import laplacian2D
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor


def initial_condotion(subkey, Grid, n1 = 10, ic='GP', length_scale_bounds=(1., 10.01)):
    """
    Function to initial fields on grid 
    """
    if 'random' in ic:
        skey1, skey2 = jax.random.split(subkey)
        U  = jax.random.uniform(skey1, shape=(Grid['nNodes'], Grid['nNodes']))
        V  = jax.random.uniform(skey2, shape=(Grid['nNodes'], Grid['nNodes']))
        X0 = jnp.stack([U,V])
    elif 'GP' in ic:
        skey1, skey2 = jax.random.split(subkey)
        # n1 = 10  # Number of points to condition on (training points)
        n2 = n1#Grid['nNodes']  # Number of points in posterior (test points)
        ny = 1  # Number of functions that will be sampled from the posterior
        domain = (jnp.min(Grid['grid_x']), jnp.max(Grid['grid_x']))

        # Sample observations (X1, y1) on the function
        X11 = jnp.linspace(domain[0], domain[1], n1)
        X12 = jnp.linspace(domain[0], domain[1], n1)
        X1  = jnp.stack(jnp.meshgrid(X11, X12, indexing='xy')).reshape(2,n1*n1).T
        y1  = jax.random.uniform(skey1, shape=(n1*n1,))
        y2  = jax.random.uniform(skey2, shape=(n1*n1,))
        # Predict points at uniform spacing to capture function
        X21 = jnp.linspace(domain[0], domain[1], n2)
        X22 = jnp.linspace(domain[0], domain[1], n2)
        X2  = jnp.stack(jnp.meshgrid(X21, X22, indexing='xy')).reshape(2,n2*n2).T


        kern = RBF(length_scale=1., length_scale_bounds=length_scale_bounds)
        gpc1 = GaussianProcessRegressor(kernel=kern, random_state=0).fit(X1, y1)
        gpc2 = GaussianProcessRegressor(kernel=kern, random_state=0).fit(X1, y2)
        # y1_mean2 = gpc1.predict(X2, return_std=False, return_cov=False)
        # y2_mean2 = gpc2.predict(X2, return_std=False, return_cov=False)
        y12 = gpc1.sample_y(X2, n_samples=ny, random_state=0)[:,0]
        y22 = gpc2.sample_y(X2, n_samples=ny, random_state=0)[:,0]
        # para= gpc.get_params()
        # std2 = jnp.sqrt(jnp.diag(y_cov2))
        # gpc.kernel_
        # X0 = jnp.stack([y1_mean2.reshape(n2,n2), y2_mean2.reshape(n2,n2)])
        X0 = jnp.stack([y12.reshape(n2,n2), y22.reshape(n2,n2)])
    return X0

def get_laplacian(Grid, UT=True):
    lap1fu = lambda t, rX: laplacian2D((rX[0][0], rX[1][0]), Grid['grid_dx'], Grid['grid_dy'], UT)
    lap2fu = lambda t, rX: laplacian2D((rX[0][1], rX[1][1]), Grid['grid_dx'], Grid['grid_dy'], UT)
    # lap1fu = lambda t, X: laplacian2D(X[0], Grid['grid_dx'], Grid['grid_dy'])
    # lap2fu = lambda t, X: laplacian2D(X[1], Grid['grid_dx'], Grid['grid_dy'])
    return lap1fu, lap2fu 

def get_sourceRxn(case, X_shape, UT=True, **args):
    """
    Function to computes the reaction source terms
    """
    # FitzHugh–Nagumo RD --------------
    if 'FN' in case:
        alpha = -0.005  if 'alpha'  not in args.keys() else args['alpha']
        beta  = 10.     if 'beta'   not in args.keys() else args['beta']
        
        # f, k = 0.055, 0.062
        # srcfu1_det = lambda t, X: - X[0]*X[1]**2 + f*(1-X[0])
        # srcfu2_det = lambda t, X: + X[0]*X[1]**2 - (k+f)*X[1]

        srcfu1_det = lambda t, X: X[0] - X[0]**3 - X[1] + alpha
        srcfu2_det = lambda t, X: beta*(X[0] - X[1])
        if UT:
            vmap_src1_det = jax.vmap(lambda X, args: srcfu1_det(args, X), in_axes=(0, None), out_axes=(0))
            vmap_src2_det = jax.vmap(lambda X, args: srcfu2_det(args, X), in_axes=(0, None), out_axes=(0))
            UT_src1       = get_2Dunsented_transform_fu(vmap_src1_det, X_shape)
            UT_src2       = get_2Dunsented_transform_fu(vmap_src2_det, X_shape)
            sourceRxn1    = lambda t, rX: UT_src1(xmean=rX[0], xcov=rX[1], args=t)
            sourceRxn2    = lambda t, rX: UT_src2(xmean=rX[0], xcov=rX[1], args=t)
        else:
            sourceRxn1    = lambda t, rX: (srcfu1_det(t, rX[0]), jnp.zeros_like(rX[0][0]))
            sourceRxn2    = lambda t, rX: (srcfu2_det(t, rX[0]), jnp.zeros_like(rX[0][0]))

    return sourceRxn1, sourceRxn2
    
    
def get_wraperNN(X_shape):

    in_fu  = lambda x: jnp.concatenate([(x[0].transpose(1,2,0)).reshape(-1,X_shape[0]),
                                        (x[1].transpose(1,2,0)).reshape(-1,X_shape[0])], axis=1)
    def out_fu(y):
        y = (y.reshape(*X_shape[1:],2)).transpose(2,0,1)
        return y[0], jax.nn.softplus(y[1]) 
    
    return in_fu, out_fu
