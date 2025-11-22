import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.ticker import StrMethodFormatter, MaxNLocator

from config.config import *
from utils.utils import *

mat_data = sio.loadmat(trajectory_file_4000)
X = mat_data["y"].T
Q = X[:, :nt]
P = X[:, nt:]

UU_train, SS_train, _ = svd(Q, full_matrices=False)

energy = np.cumsum(SS_train**2 / np.sum(SS_train**2))*100
Threshold = 99.99
energy_rank = np.argwhere(energy >= Threshold)
preserved_99 = energy_rank[0][0]
r_list = [i for i in range(2, max(preserved_99, r)+1)]

print(f"{energy[preserved_99]}% conserved at r={preserved_99}")
print(f"energy at nt>=r+r(r+1)/2: {energy[r-1]}\n")

train_errors = np.zeros(len(r_list))
pred_errors = np.zeros(len(r_list))
for i, r_rank in enumerate(r_list):
    train_errors[i] = (np.sum(SS_train[r_rank:]**2)/np.sum(SS_train**2))*100
    pred_errors[i] = ((np.linalg.norm(P - UU_train[:, :r_rank] @ UU_train[:, :r_rank].T @ P, 'fro')**2) / (np.linalg.norm(P, 'fro'))**2)*100
    if r_rank == preserved_99 or r_rank == r:
        print(f"errors at r = {r_rank}:")
        print(f"  Training: {train_errors[i]}")
        print(f"  Prediction: {pred_errors[i]}\n")

# print("Over 99.99% of energy preserved", energy_rank[0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))

fig.suptitle(f"Training and Prediction Error: nt={nt}")

ax1.plot(r_list, train_errors, label='Training error')
ax1.set_title('Training residual')
ax1.minorticks_on()
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax1.set_ylabel('% Projection error (train)')
ax1.set_xlabel('basis size $r$')
ax1.legend(prop={'size': 8})
# ax1.set_ylim(0, np.max(train_errors)*1.10)

ax2.plot(r_list, pred_errors, label='Prediction error')
ax2.set_title('Prediction residual')
ax2.minorticks_on()
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))
ax2.set_ylabel('% Projection error (prediction)')
ax2.set_xlabel('basis size $r$')
ax2.legend(prop={'size': 8})
ax2.set_ylim(0, np.max(pred_errors)*1.10)

plt.tight_layout()
plt.show()

plt.close()