import jax
import jax.numpy as jnp
import numpy as np 
import haiku as hk
from typing import Optional

# mlp model: x_i^{l+1} = x_i^l + \sum_{j \neq i} \eta(|r_ij|)*(r_ij)
class mlpmodel(hk.Module):
    def __init__(self, 
                 D :int,
                 F :int, 
                 remat: bool = False, 
                 init_stddev:float = 0.1,
                 name: Optional[str] = None
                 ):
        super().__init__(name=name)
        self.D = D
        self.F = F
        self.remat = remat
        self.init_stddev = init_stddev
  
    def phi(self, x, d):
        n, dim = x.shape 
        r_ij = (jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim)))
        r_ij = jnp.sum(jnp.square(r_ij), axis=-1).reshape(n, n, 1)
        # r_ij = r_ij ** 2
        mlp = hk.nets.MLP([self.F, 1], 
                          activation=jax.nn.softplus, 
                          w_init=hk.initializers.TruncatedNormal(0.01), 
                          b_init=hk.initializers.TruncatedNormal(0.10),
                          name=f"edge_mlp_{d}") 
        return mlp(r_ij) 

    def __call__(self, x):
        x = jnp.reshape(x, [int(x.size/2), 2])
        n, dim = x.shape
        
        def block(x, d): 
            # Calculate: sum_{j \neq i} \phi(|xi - xj|)*(xi - xj)
            m_ij = self.phi(x, d)
            weight = m_ij.reshape(n, n) / (n-1)
            r_ij = jnp.reshape(x, (n, 1, dim)) - jnp.reshape(x, (1, n, dim))
            r_ij = r_ij.reshape(n, n, dim) 
            sum_phi = jnp.einsum('ijd,ij->id', r_ij, weight)
            x = x + sum_phi
            return x
        
        if self.remat:
            block = hk.remat(block, static_argnums=2)

        for d in range(self.D):
            x = block(x, d)
            
        x = jnp.reshape(x, [x.size])
        return x