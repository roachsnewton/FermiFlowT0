import jax
import jax.numpy as jnp
import numpy as np 

# Calculate: y(x), grad(y(x)), laplacian(y(x))
def y_grad_laplacian(f, x):
    """
        Input:  x.shape (n, dim),  f(x) is a function shape(1).
        Output: y: (1),  grad_y: (n, dim),  laplacian_y: (1)
    """
    n, dim = x.shape
    # Calculate f(x)
    y = f(x)
    
    # Calculate grad_y
    grad_y = jax.grad(f)(x)
    
    # Calculate laplacian_y
    hessian_matrix = jax.hessian(f)(x)
    hessian_matrix = jnp.reshape(hessian_matrix,[n*dim, n*dim])
    laplacian_y = jnp.sum(jnp.trace(hessian_matrix))
    
    return y, grad_y, laplacian_y

def batch_y_grad_laplacian(f, x):
    '''
        Input:  x.shape (batch, n, dim),  f(x) is a function shape(1).
        Output: y: (batch),  grad_y: (batch, n, dim),  laplacian_y: (batch)
    '''
    vmap_y_grad_laplacian = jax.vmap(y_grad_laplacian, in_axes=(None, 0))
    y, grad_y, laplacian_y = vmap_y_grad_laplacian(f, x)
    return y, grad_y, laplacian_y