import os, sys
import os.path
from scipy.sparse import spdiags, lil_matrix
from scipy.integrate import solve_ivp
from scipy.linalg import svd

from utils.utils import *
from config.config import *

Xt = np.load('data/snapshots.npy')

# do an economy svd
U, S, _ = svd(Xt, full_matrices=False)

ret_energy  = np.cumsum(S**2)/np.sum(S**2)
# r           = np.argmax(ret_energy > target_energy)
Phi         = U[:, :r]

print(r)

Dxf, Dx, Lx, D3x    = diff(n, dx)  # extracting the differential operators
L3, L1, Q           = get_rom_operators(Phi, Dx, D3x, r)

# Initial Conditions
ufom0 = nsol*(nsol+1) * (1 / np.cosh(Xg))**2
urom0 = Phi.T @ ufom0

urom = urom0

uromsol         = np.zeros((r, nt + 1))  # for plotting
uromtraj        = np.zeros((n, nt + 1))  # for plotting
uromsol[:, 0]   = urom0
uromtraj[:, 0]  = Phi @ urom

for i in range(nt):
    
    # rom propagation
    sol_rom             = solve_ivp(rhsrom, (0, dt), urom, method='RK23', args=(alp, nu, rho, L3, L1, Q))
    urom                = sol_rom.y[:, -1]
    uromsol[:, i + 1]   = urom
    uromtraj[:, i + 1]  = Phi@urom

np.save(Galerkin_ROM_sol_file, uromtraj)


Qhat = Phi.T @ Xt

err = compute_train_err(Qhat.T[:nt, :], uromsol.T[:nt, :])

print(err)
print(np.linalg.norm(uromtraj[:, :nt] - Xt[:, :nt])/np.linalg.norm(Xt[:, :nt]))

for r in [5, 10, 20, 50, 75]:

    Phi         = U[:, :r]

    Dxf, Dx, Lx, D3x    = diff(n, dx)  # extracting the differential operators
    L3, L1, Q           = get_rom_operators(Phi, Dx, D3x, r)

    # Initial Conditions
    ufom0 = nsol*(nsol+1) * (1 / np.cosh(Xg))**2
    urom0 = Phi.T @ ufom0

    urom = urom0

    uromsol         = np.zeros((r, nt + 1))  # for plotting
    uromtraj        = np.zeros((n, nt + 1))  # for plotting
    uromsol[:, 0]   = urom0
    uromtraj[:, 0]  = Phi @ urom

    for i in range(nt):
        
        # rom propagation
        sol_rom             = solve_ivp(rhsrom, (0, dt), urom, method='RK23', args=(alp, nu, rho, L3, L1, Q))
        urom                = sol_rom.y[:, -1]
        uromsol[:, i + 1]   = urom
        uromtraj[:, i + 1]  = Phi@urom

    # np.save(Galerkin_ROM_sol_file, uromtraj)


    Qhat = Phi.T @ Xt

    err = compute_train_err(Qhat.T[:nt, :], uromsol.T[:nt, :])

    print('results for r = ', r)
    print(err)
    print(np.linalg.norm(uromtraj[:, :nt] - Xt[:, :nt])/np.linalg.norm(Xt[:, :nt]))