import os, sys
import os.path
from scipy.sparse import spdiags, lil_matrix
from scipy.integrate import solve_ivp

from utils.utils import *
from config.config import *

xt          = nsol * (nsol + 1) * (1 / np.cosh(Xg))**2  # sech^2(Xg)
Xt          = np.zeros((n, nt_all + 1))              # Snapshot matrix
Xt[:, 0]    = xt                                  # Initial condition

Dxf, Dx, Lx, D3x = diff(n, dx)  # extracting the differential operators

for step in range(nt_all):

    sol = solve_ivp(rhs, (0, dt), xt, method='RK45', args=(nu, alp, rho, Dx, D3x), rtol=1e-5, atol=1e-7)
    # sol = solve_ivp(rhs, (0, dt), xt, method='RK23', args=(nu, alp, rho, Dx, D3x))
    # sol = solve_ivp(rhs, (0, dt), xt, method='BDF', args=(nu, alp, rho, Dx, D3x))
    # sol = solve_ivp(rhs, (0, dt), xt, method='Radau', args=(nu, alp, rho, Dx, D3x))
    # sol = solve_ivp(rhs, (0, dt), xt, method='LSODA', args=(nu, alp, rho, Dx, D3x))

    xt              = sol.y[:, -1]
    Xt[:, step + 1] = xt

# save snapshots to file
np.save(snap_file, Xt)