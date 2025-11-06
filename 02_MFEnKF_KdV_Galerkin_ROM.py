import numpy as np
from matplotlib import pyplot as plt
import os.path
from scipy.integrate import solve_ivp
from scipy.linalg import svd
from scipy.linalg import sqrtm
from MFEnKF import *
from user_function import *

from config.config import *

Dxf, Dx, Lx, D3x = diff(n, dx)  # extracting the differential operators

if not os.path.isfile(snap_file):
    print('You need to generate the high-fidelity snapshots to train the ROM!')
    print('Run script 00_get_high_fidelity_snapshots.py')
else:
    Xt = np.load(snap_file)

Xt_train = Xt[:, :nt]

# do a economy svd
U, S, VT    = svd(Xt_train, full_matrices=False)
Phi         = U[:, :r]

L3, L1, Q = get_rom_operators(Phi, Dx, D3x, r)

# try deferencing Xt
del Xt_train

# Initial Conditions
ufom0 = nsol*(nsol+1) * (1 / np.cosh(Xg))**2
urom0 = Phi.T @ ufom0

ufom = ufom0
urom = urom0

ref_traj            = np.zeros((n, nt+1)) # for plotting
xa_Galerkin_traj    = np.zeros((n, nt+1))  # for plotting
# urom_traj = np.zeros((n, nt+1))  # for plotting

# define the observation operators and observation error covariance
# we note that there is no model error here
observe_index   = list(range(0, n, 20))
num_observation = len(observe_index)

eta     = 1
R       = np.square(eta) * np.eye(num_observation, num_observation)  # this is for the primary and control variate
sqrtR   = sqrtm(R)

R3 = 3 * R  # noise for the ancillary/rom variate

H = np.eye(n)            # Full identity matrix of size n×n
H = H[observe_index, :]

infl        = 1.01  # inflation factor for primary and control variate
infl_ROM    = 1.05  # inflation factor for ancillary variate

# create ensembles for FOM and ROM
rand_vals       = 1.5 * np.random.randn(N) # shape (N,)
Xg_col          = Xg[:, np.newaxis]           # shape (n, 1)
rand_vals_row   = rand_vals[np.newaxis, :]  # shape (1, N)
xf              = 6 * (1 / np.cosh(Xg_col - rand_vals_row))**2  # shape (n, N)

rand_vals       = 1.5 * np.random.randn(N_ROM)                 # shape: (N_ROM,)
Xg_col          = Xg[:, np.newaxis]                               # shape: (n, 1)
rand_vals_row   = rand_vals[np.newaxis, :]                 # shape: (1, N_ROM)
profiles        = 6 * (1 / np.cosh(Xg_col - rand_vals_row))**2  # shape: (n, N_ROM)
xf_ROM          = Phi.T @ profiles                                # shape: (r, N_ROM)

ref_traj[:, 0]                  = ufom
xa_Galerkin_traj[:, 0]          = np.mean(xf, axis=1)
# urom_traj[:, 0] = Phi@urom

xa                  = xf  # not needed, just to avoid confusion
xa_Galerking_ROM    = xf_ROM  # not needed, just to avoid confusion
xt                  = ufom

rmse = 0
rmse_plot = np.zeros(nt_all)
for i in range(nt_all):
    # propagate the truth
    sol = solve_ivp(rhs, [0, dt], xt, method='RK23', args=(nu, alp, rho, Dx, D3x))
    xt  = sol.y[:, -1]
    
    ref_traj[:, i+1] = xt

    # Project the FOM analysis to ROM space (control variate)
    xa_FOM_in_ROM = Phi.T @ xa  # Shape: (r, N)
    # propagate it through the rom dynamics
    for j in range(N):
        sol                 = solve_ivp(rhsrom, [0, dt], xa_FOM_in_ROM[:, j], method='RK23', args=(alp, nu, rho, L3, L1, Q))
        xa_FOM_in_ROM[:, j] = sol.y[:, -1]

    # propagate the FOM ensembles through the FOM dynamics (primary variate)
    for j in range(N):
        sol         = solve_ivp(rhs, [0, dt], xa[:, j], method='RK23', args=(nu, alp, rho, Dx, D3x))
        xa[:, j]    = sol.y[:, -1]

    # propagate the ROM through the ROM dynamics (ancillary variate)
    for j in range(N_ROM):
        sol                     = solve_ivp(rhsrom, [0, dt], xa_Galerking_ROM[:, j], method='RK23', args=(alp, nu, rho, L3, L1, Q))
        xa_Galerking_ROM[:, j]  = sol.y[:, -1]

    # create observations based around the truth (linear H for now)
    y = H@xt + sqrtR @ np.random.randn(num_observation)

    # inflating the prior covariances is done inside the MFEnKF function
    # call the MFEnKF algorithm for the filtering part
    xa, xa_Galerking_ROM        = mfenkf(xa, xa_Galerking_ROM, xa_FOM_in_ROM, Phi, y, H, R, R3, infl, infl_ROM)
    xa_mean                     = np.mean(xa, axis=1)
    xa_Galerkin_traj[:, i+1]    = xa_mean

    # you can apply spinoffs, if needed. basically we ignore the first few rmses
    rmse = np.sqrt(((np.linalg.norm(xa_mean-xt, 2))**2 + (rmse**2) * i * n)/((i+1)*n))
    # rmse = np.linalg.norm(xa_mean-xt, 2)
    rmse_plot[i] = rmse  # use this to plot, if needed

    print(f"Step = {i}, rmse = {rmse:.5f}")

# post-processing (this produces a gif)
# plt.plot(rmse_plot)
# animate_multiple_trajectories(Xg, ref_traj, xa_Galerkin_traj, dt, 'fom', 'fom_animation.gif')

np.save('data/ref_traj.npy', ref_traj)
np.save('data/xa_Galerkin_traj.npy', xa_Galerkin_traj)
np.save('data/rmse_Galerkin_traj.npy', rmse_plot)