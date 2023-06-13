import jax
import jax.numpy as jnp
import numpy as np 
from functools import partial

'''
    Markov Chain Monte Carlo sampling algorithm.
'''
# Sampling function
def mc_sample(key, flogpx, sample_shape, equilibrim_steps=100, tau=0.1):
    '''
        Input: key = jax.random.PRNGKey(xxx)
               sample_shape = [n, dim]
        Output: x (n, dim)
    '''
    
    n, dim = sample_shape
    x = jax.random.normal(key, (n, dim)) 
    logp = flogpx(x)
    
    for i in range(equilibrim_steps):
        key, subkey2, subkey3 = jax.random.split(key, num=3)

        x_new = x + tau * jax.random.normal(subkey2, x.shape)  
        logp_new = flogpx(x_new)
        p_accept = jnp.exp(logp_new - logp)
        accept = jax.random.uniform(subkey3, p_accept.shape) < p_accept
        
        x = jnp.where(accept[None, None], x_new, x)
        logp = jnp.where(accept, logp_new, logp)
    # print(i, sum(accept)/batch)
    return x

#================
# def batch_mc_sample(key, batch_flogpx, sample_shape, equilibrim_steps=100, tau=0.1):
#     '''
#         Input: key = jax.random.PRNGKey(xxx)
#                sample_shape = [batch, n, dim]
#         Output: x (batch, n, dim)
#     '''
    
#     batch, n, dim = sample_shape
#     x = jax.random.normal(key, (batch, n, dim)) 
#     logp = batch_flogpx(x)
    
#     for i in range(equilibrim_steps):
#         key, subkey2, subkey3 = jax.random.split(key, num=3)

#         x_new = x + tau * jax.random.normal(subkey2, x.shape)  
#         logp_new = batch_flogpx(x_new)
#         p_accept = jnp.exp(logp_new - logp)
#         accept = jax.random.uniform(subkey3, p_accept.shape) < p_accept
        
#         x = jnp.where(accept[:, None, None], x_new, x)
#         logp = jnp.where(accept, logp_new, logp)
        
#     acc = jnp.sum(accept)/batch
#     #print('i = ', i, 'acc_rate', jnp.sum(accept)/batch)
#     return x, acc

@partial(jax.jit, static_argnums=0)
def batch_mcmc(logp_fn, x_init, key, mc_steps = 100, mc_width = 0.1):
    def step(i, state):
        x, logp, key, num_accepts = state
        key, key_proposal, key_accept = jax.random.split(key, 3)
        
        x_proposal = x + mc_width * jax.random.normal(key_proposal, x.shape)
        logp_proposal = logp_fn(x_proposal)

        ratio = jnp.exp((logp_proposal - logp))
        accept = jax.random.uniform(key_accept, ratio.shape) < ratio

        x_new = jnp.where(accept[:, None, None], x_proposal, x)
        logp_new = jnp.where(accept, logp_proposal, logp)
        num_accepts += accept.sum()
        return x_new, logp_new, key, num_accepts
    
    logp_init = logp_fn(x_init)

    x, logp, key, num_accepts = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key, 0.))
    accept_rate = num_accepts / (mc_steps * x.shape[0])
    return x, accept_rate