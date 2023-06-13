import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np 
import haiku as hk
import optax
import matplotlib.pyplot as plt 
import time

#%%==== Import my functions  ========================================
from funpotential import batch_energy_potential
from fungradient import y_grad_laplacian
from funorbitals import f_logp
from funsampling import batch_mcmc
from funmodel import mlpmodel


#%%==== Make network ========================================
def make_network(key, n, dim, hidden_DF):
    hidden_D, hidden_F = hidden_DF
    @hk.without_apply_rng
    @hk.transform
    def network(x):
        net = mlpmodel(D=hidden_D, F=hidden_F)
        return net(x)

    x = jax.random.normal(key, (n, dim)) 
    x_reshape = jnp.reshape(x, n*dim)
    params = network.init(key, x_reshape)
    return params, network.apply

#%%==== Porbability of x: q(x)========================================
def f_logqx(x, network, params):
    ''' 
        log q(x) = log p(z) + log |\partial z/ \partial x| 
    '''
    n, dim = x.shape
    x_reshape = jnp.reshape(x, n*dim)
    z_reshape = network(params, x_reshape)
    z = jnp.reshape(z_reshape, [n, dim])
    jac = jax.jacfwd(network, argnums=1)(params, x_reshape)
    _, logabsdet = jnp.linalg.slogdet(jac)
    logqx = f_logp(z) + logabsdet
    return logqx

batch_f_logqx = jax.vmap(f_logqx, in_axes=(0, None, None))

#%%==== Make flow ========================================
def make_flow(network):
    def flow(params, x_reshape, n, dim):
        x = jnp.reshape(x_reshape, [n, dim])
        
        logp, grad_logp, laplacian_logp = y_grad_laplacian(lambda x: f_logqx(x, network, params), x)
        gradtheta_logp = jax.grad(f_logqx, argnums = 2)(x, network, params)
                
        return x, logp, grad_logp, laplacian_logp, gradtheta_logp
    return flow

#%%==== Make loss ========================================
def make_loss(batch_flow, kappa, n, dim):
    def loss(params, x_reshape):
        ''' 
            Input: params, x
            Output: loss function 
                Eloc_mean (1), Eloc_err(1), Ek_mean(1), Ep_mean(1), 
                x(batch, n, dim), 
                gradEm('dict' same as params)
        '''

        x, _, grad_logp, laplacian_logp, gradtheta_logp = batch_flow(params, x_reshape, n, dim)

        Ep = batch_energy_potential(x, kappa)
        Ek = - 1/4 * laplacian_logp - 1/8 * (grad_logp**2).sum(axis=(-2, -1))
        Eloc = Ep + Ek
        Eloc_err  = jnp.std(Eloc)/jnp.sqrt(x.shape[0])
        Eloc_mean = jnp.mean(Eloc)
        
        Eg = Eloc - Eloc_mean
        gradE = jax.tree_map(lambda g: jnp.einsum('d..., d-> d...', g, Eg), gradtheta_logp)
        gradE = jax.tree_map(lambda g: jnp.mean(g, axis=0), gradE)
        
        return Eloc_mean, Eloc_err, jnp.mean(Ek), jnp.mean(Ep), x, gradE
    return loss

#%%==== From x get z ========================================
def network_x2z(x, network, params):
    '''
        x, params -> z
    '''
    n, dim = x.shape
    x_reshape = jnp.reshape(x, n*dim)
    z_reshape = network(params, x_reshape)
    z = jnp.reshape(z_reshape, [n, dim])
    return z

batch_network_x2z = jax.vmap(network_x2z, in_axes=(0, None, None))
