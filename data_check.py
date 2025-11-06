from utils.utils import *
from config.config import *

import scipy.io as sio

mat_data = sio.loadmat(trajectory_file_500)

# print(mat_data)

Qtrain = mat_data["y"].T

print(type(Qtrain))
print(Qtrain.shape)

print(Qtrain)