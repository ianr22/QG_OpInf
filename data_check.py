from utils.utils import *
from config.config import *

import scipy.io as sio

mat_data = sio.loadmat(trajectory_file_500)

# print(mat_data)

Qtrain = mat_data["y"].T

print(type(Qtrain))
print(Qtrain.shape)

print(Qtrain)

OpInf_ROM_sol = np.load(OpInf_ROM_sol_file)

print(type(OpInf_ROM_sol))
print("Shape of op inf sol:", OpInf_ROM_sol.shape)

Galerkin_ROM_sol = np.load(Galerkin_ROM_sol_file)

print(type(Galerkin_ROM_sol))
print("Shape of Galerkin sol", Galerkin_ROM_sol.shape)