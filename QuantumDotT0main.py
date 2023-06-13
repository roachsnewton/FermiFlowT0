import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np 
import haiku as hk
import optax
import matplotlib.pyplot as plt 
import time

#%%==== import my functions  ========================================
from funsampling import batch_mcmc
from funflow import make_network, batch_f_logqx, make_flow, make_loss, batch_network_x2z

#%%==== Import paramaters  ========================================
import argparse

def parse_list(s):
    return list(map(int, s.split(',')))
parser = argparse.ArgumentParser()
parser.add_argument('--shape',       type=parse_list, default = [8192, 6, 2])
parser.add_argument('--kappa',       type=float, default = 2.00)
parser.add_argument('--hidden_DF',   type=parse_list, default = [4, 32])
parser.add_argument('--epochs_tot',  type=int, default = 1001)
parser.add_argument('--savedatas',   action='store_true', default=False)
parser.add_argument('--savefigures', action='store_true', default=False)
parser.add_argument('--fig_per_i',   type=int, default = 100)

args = parser.parse_args()
 
#%%==== Run ========================================
# python QuantumDotT0main.py --shape=8192,3,2 --kappa=0.5 --hidden_DF=2,32 --epochs_tot=1001 --fig_per_i=100 --savedatas --savefigures
# python QuantumDotT0main.py --shape=8192,3,2 --kappa=2.0 --hidden_DF=4,32 --savedatas --savefigures
# python QuantumDotT0main.py --savedatas --savefigures

#%%==== Paramaters ========================================
batch = args.shape[0]
n     = args.shape[1]
dim   = args.shape[2]
kappa = args.kappa
hidden_DF   = args.hidden_DF
epochs_tot  = args.epochs_tot
savedatas   = args.savedatas
savefigures = args.savefigures
fig_per_i   = args.fig_per_i

#%%==== Initial ========================================
key = jax.random.PRNGKey(42)

params, network = make_network(key, n, dim, hidden_DF)
flow = make_flow(network)
batch_flow = jax.vmap(flow, in_axes=(None, 0, None, None))
loss = make_loss(batch_flow, kappa, n, dim)

#%%==== Opmization ========================================
optimizer = optax.adam(0.001)
opt_state = optimizer.init(params)

#%%==== Training ========================================
@jax.jit
def step(x, params, opt_state):
    value = loss(params, jnp.reshape(x, [batch, n*dim]))
    updates, opt_state = optimizer.update(value[-1], opt_state)
    params = optax.apply_updates(params, updates)
    return value, params, opt_state

t_init = time.time()
dy = []
loss_history = []

print("================================================================================")
print("shape ", [batch, n, dim],
        ", kappa", kappa,
        ", hidden", hidden_DF,
        ", epochs", epochs_tot,
        ", params:", jax.flatten_util.ravel_pytree(params)[0].size)
print("--------------------------------------------------------------------------------")

#%%==== Main Program ========================================
for i in range(epochs_tot):
    t0 = time.time()
    
    key, subkey = jax.random.split(key)
    
    x_init = jax.random.normal(subkey, [batch, n, dim]) 
    x, acc = batch_mcmc(lambda x: batch_f_logqx(x, network, params), x_init, 
                        subkey, mc_steps = 2000, mc_width = 0.1)
    
    value, params, opt_state = step(x, params, opt_state)
    
    Eloc_mean, Eloc_err, Ek_mean, Ep_mean, x, gradE = value
    loss_history.append([Eloc_mean, Eloc_err, Ek_mean, Ep_mean, acc])
    
    if i%fig_per_i == 0 and savefigures:
        fig = plt.figure(figsize=(16, 5))
        
        z = batch_network_x2z(x, network, params)
    
        x0 = jnp.reshape(x, (batch*n, dim)) 
        ## figure 1
        plt.subplot(1, 3, 1, aspect=1)
        plt.title(['x epochs = ', i])
        H, xedges, yedges = np.histogram2d(x0[:, 0], x0[:, 1], bins=100, 
                                           range=((-5, 5), (-5, 5)), density=True)
        plt.imshow(H, interpolation="nearest", 
                   extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), cmap="inferno")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])

        z0 = jnp.reshape(z, (batch*n, dim)) 
        ## figure 2
        plt.subplot(1, 3, 2, aspect=1)
        plt.title(['z epochs = ', i])
        H, xedges, yedges = np.histogram2d(z0[:, 0], z0[:, 1], bins=100, 
                                           range=((-5, 5), (-5, 5)), density=True)
        plt.imshow(H, interpolation="nearest", 
                   extent=(xedges[0], xedges[-1], yedges[0], yedges[-1]), cmap="inferno")
        plt.xlim([-5, 5])
        plt.ylim([-5, 5])
        
        ## figure 3
        plt.subplot(1, 3, 3)
        plt.title(['E =', "{:.3f}".format(Eloc_mean), 'err =', "{:.3f}".format(Eloc_err)])
        px = np.arange(i+1)
        py = np.array(loss_history)
        plt.plot(px, py[:, 0], marker='.')
        plt.xlabel('epochs')
        plt.ylabel('variational free energy')
        ## savefig
        figurename = "./DatasT0_hidden%d_%d" %(hidden_DF[0], hidden_DF[1]) \
                    + "_n%d" % (n)+ "_k%.2f" % (kappa) + "_i%d" % (i) + ".jpg"
        plt.savefig(figurename)
        plt.close('all')

    t1 = time.time()

    print(i, 
          ". acc =",  "{:.3f}".format(acc),
          ", E =",    "{:.3f}".format(Eloc_mean), 
          ", err =",  "{:.3f}".format(Eloc_err), 
          ", Ek =",   "{:.3f}".format(Ek_mean), 
          ", Ep =",   "{:.3f}".format(Ep_mean), 
          ", dt =",   "{:.3f}".format(t1-t0),
          ", t =",    "{:.3f}".format(t1-t_init))

    
#%% Save datas ========================================

if savedatas:
    filename = "./DatasT0_hidden%d_%d" %(hidden_DF[0], hidden_DF[1]) \
            + "_n%d" % (n)+ "_k%.2f" % (kappa) + ".npz"
    np.savez(filename,
            sample_shape = [batch, n, dim],
            hidden_DF = hidden_DF,
            kappa = kappa,
            loss_history = loss_history,
            x = x,
            z = z,
            params = params)

