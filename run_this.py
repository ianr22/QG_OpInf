# import all the necessary packages
import os, sys
import os.path
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp
from user_function import *
from scipy.linalg import svd

nu = -1.0
alp = -3.0
rho = 0.0

nsol        = 2
pod_modes   = 50

n   = 1000  # this is the number of discrete points
Xg  = np.linspace(-10, 10, n)
dx  = Xg[1] - Xg[0]

dt          = 1e-3  # step size
# nt_all        = 5000  # number of time steps
# nt            = 2500 

nt_all      = 2000  # number of time steps
nt          = 2000 
snap_shots  = 2000  # for creating snapshot matrix

xt = nsol * (nsol + 1) * (1 / np.cosh(Xg))**2  # sech^2(Xg)
Xt = np.zeros((n, snap_shots + 1))              # Snapshot matrix
Xt[:, 0] = xt                                  # Initial condition

Dxf, Dx, Lx, D3x = diff(n, dx)  # extracting the differential operators

if os.path.isfile('snapshot.npy'):
    Xt = np.load('snapshot.npy')

if not os.path.isfile('snapshot.npy'):
    for step in range(snap_shots):

        sol = solve_ivp(rhs, (0, dt), xt, method='RK45', args=(nu, alp, rho, Dx, D3x), rtol=1e-5, atol=1e-7)
        # sol = solve_ivp(rhs, (0, dt), xt, method='RK23', args=(nu, alp, rho, Dx, D3x))
        # sol = solve_ivp(rhs, (0, dt), xt, method='BDF', args=(nu, alp, rho, Dx, D3x))
        # sol = solve_ivp(rhs, (0, dt), xt, method='Radau', args=(nu, alp, rho, Dx, D3x))
        # sol = solve_ivp(rhs, (0, dt), xt, method='LSODA', args=(nu, alp, rho, Dx, D3x))

        xt = sol.y[:, -1]

        Xt[:, step + 1] = xt

        # save for future use
        np.save('snapshot.npy', Xt)

# do a economy svd
U, S, VT = svd(Xt, full_matrices=False)
Phi = U[:, :pod_modes]

L3, L1, Q = get_rom_operators(Phi, Dx, D3x, pod_modes)

# Initial Conditions
ufom0 = nsol*(nsol+1) * (1 / np.cosh(Xg))**2
urom0 = Phi.T @ ufom0

ufom = ufom0
urom = urom0

ufomtraj = np.zeros((n, nt+1))  # for plotting
uromtraj = np.zeros((n, nt+1))  # for plotting

ufomtraj[:, 0] = ufom
uromtraj[:, 0] = Phi@urom

for i in range(nt):
    # fom propagation
    sol_fom = solve_ivp(rhs, (0, dt), ufom, method='RK23', args=(nu, alp, rho, Dx, D3x))
    ufom = sol_fom.y[:, -1]
    ufomtraj[:, i+1] = ufom

    # rom propagation
    sol_rom = solve_ivp(rhsrom, (0, dt), urom, method='RK23', args=(alp, nu, rho, L3, L1, Q))
    urom = sol_rom.y[:, -1]
    uromtraj[:, i+1] = Phi@urom

    print(i)

# np.save('fom_traj.npy', ufomtraj)
# np.save('rom_traj.npy', uromtraj)

plot_trajectory(Xg, Xt, dt, 'fom_snapshot', 'fom_snapshot.gif')
plot_trajectory(Xg, ufomtraj, dt, 'fom', 'fom_animation.gif')
plot_trajectory(Xg, uromtraj, dt, 'rom', 'rom_animation.gif')




