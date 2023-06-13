import jax
import jax.numpy as jnp
import numpy as np 


'''
    36 orbitals: orbitals[0] ~ orbitals[35]
    E_indices 
        E(0): 0
        E(1): 1  2
        E(2): 3  4  5
        E(3): 6  7  8  9
        E(4): 10 11 12 13 14
        E(5): 15 16 17 18 19 20
        E(6): 21 22 23 24 25 26 27
        E(7): 28 29 30 31 32 33 34 35
'''

pi_sqrt_inverse = 1. / np.sqrt(np.pi)
orbitals_1d = [
    lambda x: 1,
    lambda x: np.sqrt(2) * x,
    lambda x: 1 / np.sqrt(2) * (2*x**2 - 1),
    lambda x: 1 / np.sqrt(3) * (2*x**3 - 3*x),
    lambda x: 1 / np.sqrt(6) * (2*x**4 - 6*x**2 + 1.5),
    lambda x: 1 / np.sqrt(15) * (2*x**5 - 10*x**3 + 7.5*x),
    lambda x: 1 / np.sqrt(5) * (2/3*x**6 - 5*x**4 + 7.5*x**2 - 1.25),
    lambda x: 1 / np.sqrt(70) * (4/3*x**7 - 14*x**5 + 35*x**3 - 17.5*x),
    ]
orbital_2d = lambda nx, ny: lambda x: \
                pi_sqrt_inverse * jnp.exp(- 0.5 * jnp.sum(x**2, axis=-1)) \
                * orbitals_1d[nx](x[..., 0]) \
                * orbitals_1d[ny](x[..., 1])

orbitals = [orbital_2d(nx, n - nx) for n in range(8) for nx in range(n + 1)]
#Es = [n + 1 for n in range(8) for nx in range(n + 1)]
#E_indices = lambda n: tuple(range(n*(n+1)//2, (n+1)*(n+2)//2))

def orbitals2D():
    return orbitals

#==================================================================
# Function: Log Abs Det of 1/sqrt(n!)*|\phi_i(x_j))|
def slater_logabsdet(orbitals, x):
    '''
        Only satisfy n = 1, 3, 6, 10
        Input:  x (n, dim)
        Calculate: D (n, n)
        Output: logabsdet (1)
    '''
    n, dim = x.shape
    if n == 1:
        D = jnp.array([orbitals[0](x)])
    elif n == 3:
        D = jnp.array([orbitals[0](x), orbitals[1](x), orbitals[2](x)])
    elif n == 6:
        D = jnp.array([orbitals[0](x), orbitals[1](x), orbitals[2](x), 
                       orbitals[3](x), orbitals[4](x), orbitals[5](x)])
    elif n == 10:
        D = jnp.array([orbitals[0](x), orbitals[1](x), orbitals[2](x), orbitals[3](x), orbitals[4](x), 
                       orbitals[5](x), orbitals[6](x), orbitals[7](x), orbitals[8](x), orbitals[9](x)])
    else:
        print('Error with n')
        
    D = 1 / np.sqrt(np.math.factorial(n)) * D
    _, logabsdet = jnp.linalg.slogdet(D)
    return logabsdet

def batch_slater_logabsdet(orbitals, x):
    '''
        Input:  x (batch, n, dim)
        Output: logabsdet (batch)
    '''
    vmap_logabsdet = jax.vmap(slater_logabsdet, in_axes=(None, 0))
    return vmap_logabsdet(orbitals, x)

# Function:
# log|Psi(x)| = slater_logabsdet(x)
# log(p) = 2 * log|Psi(x)|
def f_logp(x):
    fy = slater_logabsdet(orbitals, x) * 2
    return fy


def batch_f_logp(x):
    '''
    Input:  x(batch, n, dim)
    Output: logabsdet(batch)
    '''
    fy = batch_slater_logabsdet(orbitals, x) * 2
    return fy