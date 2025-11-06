from matplotlib.pyplot import *

from config.config import *

rmse_POD_Galerkin_ROM   = np.load('data/rmse_Galerkin_traj.npy')
rmse_OpInf_ROM          = np.load('data/rmse_OpInf_traj.npy')

fontsize = 12

rc("figure", dpi=400)           # High-quality figure ("dots-per-inch")
rc("text", usetex=True)         # Crisp axis ticks
rc("font", family="sans-serif")      # Crisp axis labels
# rc("legend", edgecolor='none')  # No boxes around legends
rc('text.latex', preamble=r'\usepackage{amsfonts}')
rcParams["figure.figsize"] = (9, 5)
rcParams.update({'font.size': fontsize})

# line settings for white base
charcoal    = [0.0, 0.0, 0.0]
color1      = '#d95f02'
color2      = '#7570b3'

# white base settings
rc("figure",facecolor='w')
rc("axes",facecolor='w',edgecolor=charcoal,labelcolor=charcoal)
rc("savefig",facecolor='w')
rc("text",color=charcoal)
rc("xtick",color=charcoal)
rc("ytick",color=charcoal)
 
fig, ax     = subplots()

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

x = np.arange(0, nt)

ax.plot(x, rmse_POD_Galerkin_ROM, linestyle='-', lw=1.2, color=color2, label='RMSE MFEnKF with POD Galerkin ROM (r = {})'.format(r))
ax.plot(x, rmse_OpInf_ROM, linestyle='-', lw=1.2, color=color1, label='RMSE MFEnKF with OpInf ROM (r = {})'.format(r))

ax.set_xlabel('time iterations')
ax.set_ylabel('RMSE')
ax.legend(loc='best', ncol=1)

tight_layout()

savefig('figures/MFEnKF_RMSE_ROMs.png',pad_inches=3)

close()