import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde
import numpy as np

def traj(traj, ninterleaves, gridsize=50, vmax=40, logscale=1, mincnt=1):
    """
    Visualize kspace trajectory.
    
    Args:
    traj (ndarray): kspace trajectory
    ninterleaves (int): number of interleaves
    gridsize (int): hexbin gridsize
    vmax (float): hexbin upper axis limit
    logscale (bool): if True, hexbin will be logscale
    mincnt (int): minimum number of data point per hex in hexbin
    """

    # Reshape to resolve interleaves in a new dimension
    traj_reshaped = np.reshape(traj, [int(len(traj)/ninterleaves), ninterleaves, 2], 'F')
    
    fig, axs = plt.subplots(2, 2, dpi=300)
    
    # --------- Trajectory ------------------------
    axs[0, 0].scatter(traj_reshaped[:,0,0], traj_reshaped[:,0,1], s=0.15)

    # --------- Kernel Density Estimation ---------
    # Create a grid to evaluate the density
    xi, yi = np.mgrid[min(traj_reshaped[:,0,0]):max(traj_reshaped[:,0,0]):100j,
                      min(traj_reshaped[:,0,1]):max(traj_reshaped[:,0,1]):100j]
    kde = gaussian_kde(np.vstack([traj_reshaped[:,0,0], traj_reshaped[:,0,1]])) # Compute the KDE
    density = kde(np.vstack([xi.flatten(), yi.flatten()])) # Evaluate the density on the grid

    # Plot the Density
    #plt.figure(figsize=(10, 8))
    I = axs[0, 1].imshow(density.reshape(xi.shape), origin='lower', aspect='auto',
                     extent=[min(traj_reshaped[:,0,0]), max(traj_reshaped[:,0,0]),
                             min(traj_reshaped[:,0,1]), max(traj_reshaped[:,0,1])], cmap='jet')

    fig.colorbar(I, ax=axs[0, 1], label='Counts in bin')
    
    # --------- 1D Traj ---------------------------
    axs[1, 0].plot(traj_reshaped[:,0,0], linewidth=0.8)
    axs[1, 0].plot(traj_reshaped[:,0,1], linewidth=0.8)
    
    # --------- Hexbin ----------------------------
    if logscale:
        hb = axs[1, 1].hexbin(traj[:,0], traj[:,1], gridsize=gridsize,
                              cmap='jet', mincnt=mincnt, norm=LogNorm(vmax=vmax))
    else:
        hb = axs[1, 1].hexbin(traj[:,0], traj[:,1], gridsize=gridsize,
                              cmap='jet', mincnt=mincnt, vmax=vmax)
    fig.colorbar(hb, ax=axs[1, 1], label='Counts in bin')

    for ax in axs.flat:
        #ax.set_aspect('equal')#, adjustable='box')
        ax.set_box_aspect(1)
        
    plt.tight_layout()
    plt.show()