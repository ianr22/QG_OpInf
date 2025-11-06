import numpy as np
from itertools import product
from time import time

# might want to fix rng for numpy for reproducibility
np.random.seed(16559)

# this script has a sequential data assimilation for kdv
nu      = -1.0
alp     = -3.0
rho     = 0.0

nsol        =  2
r   		= 33

N       = 25
N_ROM   = 100

n   = 63*127  # this is the number of discrete points
Xg  = np.linspace(-10, 10, n)
dx  = Xg[1] - Xg[0]

dt          = 0.0109  # step size
nt_all      = 500 # total number of snapshots
nt          = 500 # the end of the training time horizon

target_energy = 0.9999

B1 = np.logspace(-5., 2., num=10)
B2 = np.logspace(-2., 4., num=10)

# B1 = np.logspace(-5., 2., num=10)
# B2 = np.logspace(0., 5., num=10)

max_growth = 1.2

CENTERING = False

snap_file = './data/snapshots.npy'
trajectory_file_4000 = './data/trajectory_for_ROM_4000.mat'
trajectory_file_500 = './data/trajectory_for_ROM_500.mat'

Galerkin_ROM_sol_file 	= './data/Galerkin_ROM_sol.npy'
OpInf_ROM_sol_file 		= './data/OpInf_ROM_sol.npy'
