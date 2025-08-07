import jax
import jax.numpy as jnp
from jax import lax
from math import prod

from utils.utils import dummy_scan_fu

# from haiku._src.recurrent import LSTMState
# from nn.CLSTM import initial_state

def get_layer_solver(f, lsolver='fwdc', tol=1e-6, length=1000):

    def fwd_solver(f, z_init):
        z_star = dummy_scan_fu(f, init=z_init, length=length)
        return z_star
        
    def fwdc_solver(f, z_init):

        def cond_fun(carry):
            z_prev, z = carry
            return jnp.linalg.norm(z_prev['vx'] - z['vx'])/prod(z['vx'].shape[-2:]) > tol

        def body_fun(carry):
            _, z = carry
            return z, f(z)

        init_carry = (z_init, f(z_init))
        _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
        return z_star

    def newton_solver(f, z_init):
        f_root = lambda z: f(z) - z
        # g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
        def g(z): 
            f_root_ = lambda z, i: f_root(z)[i]
            for i in range(z.shape[0]):
                z - jnp.linalg.solve(jax.jacobian(f_root_)(z[i:i+1], i), f_root(z[i:i+1])[i])
        return fwd_solver(g, z_init)

    def anderson_solver(f, z_init, m=5, lam=1e-4, max_iter=length, tol=1e-5, beta=1.0):
        x0 = z_init
        x1 = f(x0, None)[0]
        x2 = f(x1, None)[0]
        X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
        F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

        res = []
        for k in range(2, max_iter):
            n = min(k, m)
            G = F[:n] - X[:n]
            GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
            H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],
                        [ jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
            alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]

            xk = beta * jnp.dot(alpha, F[:n]) + (1-beta) * jnp.dot(alpha, X[:n])
            X = X.at[k % m].set(xk)
            F = F.at[k % m].set(f(xk, None)[0])

            res = jnp.linalg.norm(F[k % m] - X[k % m]) / (1e-5 + jnp.linalg.norm(F[k % m]))
            if res < tol:
                break
        return xk

    if 'fwd' == lsolver:
        return lambda z: fwd_solver(f,z)
    if 'fwdc' == lsolver:
        return lambda z: fwdc_solver(f,z)
    elif 'newton' == lsolver:
        return lambda z: newton_solver(f,z)
    elif 'anderson' == lsolver:
        return lambda z: anderson_solver(f,z)
        
class Fp_Adjoint():

    def __init__(self, f_rhs, length, lsolver='fwdc', tol=[1e-6, 1e-6]) -> None:
        self.lsolver  = lsolver
        self.f_rhs  = f_rhs
        self.length = length
        self.tol = tol

    def fp_solver_fwd(self, params, data, **args):

        z_init   = {'vx':data['vx']}
        b, nx,ny = z_init['vx'].shape
        z_init['vx'] = z_init['vx'].reshape(b,nx*ny)

        f_z = lambda z: self.f_rhs(params, data, z)
        
        z_star = get_layer_solver(f_z, self.lsolver, **args)(z_init)
        return z_star

    def add_adjoint_backprop(self):
        
        @jax.custom_vjp
        def fp_layer(params, data):
            z_star = self.fp_solver_fwd(params, data, tol=self.tol[0], length=self.length)
            return z_star

        def fp_layer_fwd(params, data):
            z_star = fp_layer(params, data)
            return z_star, (params, data, z_star)

        def fp_layer_bwd(res, z_star_bar):
            params, data, z_star = res

            _, vjp_a = jax.vjp(lambda params, data: self.f_rhs(params, data, z_star), params, data)
            _, vjp_z = jax.vjp(lambda z: self.f_rhs(params, data, z), z_star)

            u_f = lambda u: jax.tree_util.tree_map(lambda vjp_z, z_star: vjp_z+z_star, vjp_z(u)[0], z_star_bar) 
            uT = get_layer_solver(u_f, self.lsolver, tol=self.tol[1], length=self.length)(jax.tree_util.tree_map( lambda z: jnp.zeros_like(z), z_star))

            return vjp_a(uT)

        fp_layer.defvjp(fp_layer_fwd, fp_layer_bwd)

        return fp_layer
    
class Fp_Adjoint1():

    def __init__(self, get_fp, vkeys, length=1000, lsolver='fwdc', tol=[1e-6, 1e-6]) -> None:
        self.lsolver  = lsolver
        self.get_fp = get_fp
        self.vkeys = vkeys
        self.length = length
        self.tol = tol

    def fp_solver_fwd(self, params, data):
        z_init = {}
        for vkey in self.vkeys:
            z_init[vkey] = data[vkey]
        F_fp   = self.get_fp(params, data)
        f_z = lambda z: F_fp(params, data, z)
        z_star = get_layer_solver(f_z, self.lsolver, tol=self.tol[0], length=self.length)(z_init)
        return z_star

    def add_adjoint_backprop(self):
        
        @jax.custom_vjp
        def fp_layer(params, data):
            z_star = self.fp_solver_fwd(params, data)
            return z_star

        def fp_layer_fwd(params, data):
            z_star = fp_layer(params, data)
            return z_star, (params, data, z_star)

        def fp_layer_bwd(res, z_star_bar):
            params, data, z_star = res
            F_fp   = self.get_fp(params, data)

            _, vjp_a = jax.vjp(lambda params, data: F_fp(params, data, z_star), params, data)
            _, vjp_z = jax.vjp(lambda z: F_fp(params, data, z), z_star)

            u_f = lambda u: jax.tree_util.tree_map(lambda vjp_z, z_star: vjp_z+z_star, vjp_z(u)[0], z_star_bar)
            uT = get_layer_solver(u_f, self.lsolver, tol=self.tol[1], length=self.length)(jax.tree_util.tree_map( lambda z: jnp.zeros_like(z), z_star))

            return vjp_a(uT)

        fp_layer.defvjp(fp_layer_fwd, fp_layer_bwd)

        return fp_layer

    


class lin_Adjoint():

    def __init__(self, solver, get_Ax, get_b, tol=[1e-6, 1e-6]) -> None:
        self.solver = solver
        self.get_Ax = get_Ax
        self.get_b = get_b
        self.tol = tol

    def linear_solve_fwd(self, params,data, tol=1e-6):

        Ax_fu = self.get_Ax(params=params, data=data)
        b     = self.get_b (params=params, data=data)

        solve_zb = lambda z, b: self.solver(Ax_fu, b, init=z, tol=tol)  
        return solve_zb(b, b)


    def add_adjoint_backprop(self):
        # @partial(jax.custom_vjp, nondiff_argnums=(0))
        @jax.custom_vjp
        def f(params, data):
            z_star = self.linear_solve_fwd(params,data, tol=self.tol[0])
            return z_star

        def adj_AX_fwd(params, data):
            z_star = f(params, data)
            return z_star, (params, data, z_star[0])

        def adj_AX_bwd(res, z_star_bar):
            params, data, z_star = res
            fu_zero = lambda p,D,z: jax.tree_util.tree_map(lambda Ax,b: Ax-b, 
                                                           self.get_Ax(params=p, data=D)(z), 
                                                           self.get_b(params=p, data=D))

            _, vjp_a = jax.vjp(lambda params, data: fu_zero(params, data, z_star), params, data)
            _, vjp_z = jax.vjp(lambda z: fu_zero(params, data, z), z_star)

            uT_df_fu = lambda uT: vjp_z(uT)[0]
            wT = z_star_bar[0]
            solve_zb = lambda init, b: self.solver(uT_df_fu, b, init=init, tol=self.tol[1])
            # solve_zb = lambda z, b: self.solver(uT_df_fu, b, x0=z, tol=self.tol[1])[0]
            uT, adj_info = solve_zb(jax.tree_util.tree_map( lambda wT: jnp.zeros_like(wT), wT), jax.tree_util.tree_map( lambda wT: -wT, wT))

            return vjp_a(uT)

        f.defvjp(adj_AX_fwd, adj_AX_bwd)

        return f
    
class lin_Adjoint1():

    def __init__(self, solver, get_Ax, get_b, tol=[1e-6, 1e-6], maxiter=None) -> None:
        self.solver = solver
        self.get_Ax = get_Ax
        self.get_b = get_b
        self.tol = tol
        self.maxiter = maxiter

    def linear_solve_fwd(self, params,data,data_pret, tol=1e-6):

        Ax_fu = self.get_Ax(params=params, data=data)
        b     = self.get_b (params=params, data=data_pret)

        if self.maxiter:
            solve_zb = lambda z, b: self.solver(Ax_fu, b, init=z, tol=tol, maxiter=self.maxiter)
        else: 
            solve_zb = lambda z, b: self.solver(Ax_fu, b, init=z, tol=tol)  

        return solve_zb(b, b)


    def add_adjoint_backprop(self):
        # @partial(jax.custom_vjp, nondiff_argnums=(0))
        @jax.custom_vjp
        def f(params, data, data_pret):
            z_star = self.linear_solve_fwd(params,data,data_pret, tol=self.tol[0])
            return z_star

        def adj_AX_fwd(params, data, data_pret):
            z_star = f(params, data, data_pret)
            return z_star, (params, data, data_pret, z_star)

        def adj_AX_bwd(res, z_star_bar):
            params, data, data_pret, z_star = res
            fu_zero = lambda p,D,Dt,z: jax.tree_util.tree_map(lambda Ax,b: Ax-b,
                                                              self.get_Ax(params=p, data=D)(z),
                                                              self.get_b(params=p, data=Dt))

            _, vjp_a = jax.vjp(lambda params, data, data_pret: fu_zero(params, data, data_pret, z_star), params, data, data_pret)
            _, vjp_z = jax.vjp(lambda z: fu_zero(params, data, data_pret, z), z_star)

            uT_df_fu = lambda uT: vjp_z(uT)[0]
            wT = z_star_bar
            solve_zb = lambda init, b: self.solver(uT_df_fu, b, init=init, tol=self.tol[1])
            # solve_zb = lambda z, b: self.solver(uT_df_fu, b, x0=z, tol=self.tol[1])[0]
            uT = solve_zb(jax.tree_util.tree_map( lambda wT: jnp.zeros_like(wT), wT), jax.tree_util.tree_map( lambda wT: -wT, wT))

            return vjp_a(uT)

        f.defvjp(adj_AX_fwd, adj_AX_bwd)

        return f
