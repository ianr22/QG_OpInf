import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.ticker import StrMethodFormatter

from config.config import *
from utils.utils import *

nEigs = 30
idx = [i for i in range(nEigs)]

mat_data = sio.loadmat(trajectory_file_500)
X = mat_data["y"].T

UU, SS, _ = svd(X, full_matrices=False)

# nList = [4*(i+1) for i in range(15)]
# errU2r = np.zeros(len(nList))
# for i,n in enumerate(nList):
#     U       = UU[:,:n]
#     reconU2  = U @ U.T @ X
#     errU2r[i] = relError(X, reconU2)

energy = np.cumsum(SS[:nEigs] / np.sum(SS))*100
print("Energy at r=20:", energy[19])
print("Energy at r=30:", energy[-1])
Threshold = 99.99
energy_rank = np.argwhere(energy >= Threshold)
# print("Over 99.99% of energy preserved", energy_rank[0])

fig, ax = plt.subplots()
ax.scatter(idx, energy, s=10, label='Ordinary POD (no MC)')
ax.set_title('POD Snapshot Energy')
# ax[0].set_yscale('log')
# ax[1].semilogy(nList, errU2r, label='Ordinary POD (no MC)', marker='o', linestyle='-', markersize=5)
# ax[1].set_title('POD Projection Error')
# ax[1].set_ylabel('relative $L^2$ error')
# ax[0].get_shared_y_axes().join(ax[0], ax[1])
# ax[1].set_xticklabels([])
# for i in range(1):
ax.minorticks_on()
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax.set_ylabel('% POD energy')
# for i in range(2):
ax.set_xlabel('basis size $n$')
ax.legend(prop={'size': 8})
plt.tight_layout()
# plt.savefig('KdVpodEnergy')
plt.show()