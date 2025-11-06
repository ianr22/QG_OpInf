import numpy as np
from matplotlib import pyplot as plt
import os.path
from scipy.integrate import solve_ivp
from scipy.linalg import svd
from scipy.linalg import sqrtm
from MFEnKF import *
from user_function import *
import scipy.io as sio

from utils.utils import *
from config.config import *

data = np.load('data/OpInf_ROM_operators.npz')

# Ahat1   = data['Ahat1']
# Ahat2   = data['Ahat2']
# Fhat    = data['Fhat']

Ahat   = data['Ahat']
Fhat   = data['Fhat']

# OpInf_red_model    = lambda x: nu*Ahat1 @ x + rho*Ahat2 @ x + alp*Fhat @ compute_Qhat_sq(x)

OpInf_red_model    = lambda x: Ahat @ x + Fhat @ compute_Qhat_sq(x)

Dxf, Dx, Lx, D3x = diff(n, dx)  # extracting the differential operators

if os.path.isfile(trajectory_file_500):
    mat_data = sio.loadmat(trajectory_file_500)
    Xt = mat_data["y"].T

if not os.path.isfile(trajectory_file_500):
    for step in range(nt_all):

        sol = solve_ivp(rhs, (0, dt), xt, method='RK45', args=(nu, alp, rho, Dx, D3x), rtol=1e-5, atol=1e-7)
        # sol = solve_ivp(rhs, (0, dt), xt, method='RK23', args=(nu, alp, rho, Dx, D3x))
        # sol = solve_ivp(rhs, (0, dt), xt, method='BDF', args=(nu, alp, rho, Dx, D3x))
        # sol = solve_ivp(rhs, (0, dt), xt, method='Radau', args=(nu, alp, rho, Dx, D3x))
        # sol = solve_ivp(rhs, (0, dt), xt, method='LSODA', args=(nu, alp, rho, Dx, D3x))

        xt = sol.y[:, -1]

        Xt[:, step + 1] = xt

        # save for future use
        np.save('data/snapshots.npy', Xt)

# do a economy svd
Phi = np.load('data/OpInf_POD_basis.npy')

# try deferencing Xt
del Xt

# Initial Conditions
ufom0 = nsol*(nsol+1) * (1 / np.cosh(Xg))**2
urom0 = Phi.T @ ufom0

ufom = ufom0
urom = urom0

ref_traj            = np.zeros((n, nt+1)) # for plotting
xa_OpInf_traj       = np.zeros((n, nt+1))  # for plotting
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
xa_OpInf_traj[:, 0]          = np.mean(xf, axis=1)
# urom_traj[:, 0] = Phi@urom

xa                  = xf  # not needed, just to avoid confusion
xa_OpInf_ROM       = xf_ROM  # not needed, just to avoid confusion
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
        # sol                 = solve_ivp(rhsrom, [0, dt], xa_FOM_in_ROM[:, j], method='RK23', args=(alp, nu, rho, L3, L1, Q))
        xa_FOM_in_ROM[:, j] = OpInf_red_model(xa_FOM_in_ROM[:, j])

    # propagate the FOM ensembles through the FOM dynamics (primary variate)
    for j in range(N):
        sol         = solve_ivp(rhs, [0, dt], xa[:, j], method='RK23', args=(nu, alp, rho, Dx, D3x))
        xa[:, j]    = sol.y[:, -1]

    # propagate the ROM through the ROM dynamics (ancillary variate)
    for j in range(N_ROM):
        # sol                     = solve_ivp(rhsrom, [0, dt], xa_OpInf_ROM[:, j], method='RK23', args=(alp, nu, rho, L3, L1, Q))
        xa_OpInf_ROM[:, j]  = OpInf_red_model(xa_OpInf_ROM[:, j])

    # create observations based around the truth (linear H for now)
    y = H@xt + sqrtR @ np.random.randn(num_observation)

    # inflating the prior covariances is done inside the MFEnKF function
    # call the MFEnKF algorithm for the filtering part
    xa, xa_OpInf_ROM        = mfenkf(xa, xa_OpInf_ROM, xa_FOM_in_ROM, Phi, y, H, R, R3, infl, infl_ROM)
    xa_mean                     = np.mean(xa, axis=1)
    xa_OpInf_traj[:, i+1]    = xa_mean

    # you can apply spinoffs, if needed. basically we ignore the first few rmses
    rmse = np.sqrt(((np.linalg.norm(xa_mean-xt, 2))**2 + (rmse**2) * i * n)/((i+1)*n))
    # rmse = np.linalg.norm(xa_mean-xt, 2)
    rmse_plot[i] = rmse  # use this to plot, if needed

    print(f"Step = {i}, rmse = {rmse:.5f}")

# post-processing (this produces a gif)
# plt.plot(rmse_plot)
# animate_multiple_trajectories(Xg, ref_traj, xa_OpInf_traj, dt, 'fom', 'fom_animation.gif')

# np.save('data/ref_traj.npy', ref_traj)
np.save('data/xa_OpInf_traj.npy', xa_OpInf_traj)
np.save('data/rmse_OpInf_traj.npy', rmse_plot)