import jax
import jax.numpy as jnp
import numpy as np 

# Calcualte energy
def energy_potential(x, kappa):
    '''
        Input:     x.shape (n, dim)
        Calculate: V = 0.5 * \sum_i x_i^2 + \sum_{i<j}^n kappa / v(|r_i - r_j|)
        Output:    V.shape (1)
    '''
    n, dim = x.shape
    i, j = np.triu_indices(n, k=1)
    r_ij = jnp.linalg.norm((jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))[i,j], axis=-1)
    v_ee = jnp.sum(kappa / r_ij)
    return 0.5 * jnp.sum(x**2) + v_ee

def batch_energy_potential(x, kappa):
    '''
        Input:     x.shape (batch, n, dim)
        Calculate: V = 0.5 * \sum_i x_i^2 + \sum_{i<j}^n kappa / v(|r_i - r_j|)
        Output:    V.shape (batch)
    '''
    return jax.vmap(energy_potential, (0, None), 0)(x, kappa)