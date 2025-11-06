from matplotlib.pyplot import *
import matplotlib.animation as animation

from config.config import *

FOM_data = np.load('data/ref_traj.npy')

Galerkin_ROM_data   = np.load(Galerkin_ROM_sol_file)
OpInf_ROM_data      = np.load(OpInf_ROM_sol_file)

print(FOM_data.shape)

fontsize = 5

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

line1,      = ax.plot(Xg, FOM_data[:, 0], lw=1.0, color=charcoal, label='FOM')
line2,      = ax.plot(Xg, Galerkin_ROM_data[:, 0], lw=0.75, color=color2, label='POD Galerkin ROM (r = {})'.format(r))
line3,      = ax.plot(Xg, OpInf_ROM_data[:, 0], lw=0.75, color=color1, label='OpInf ROM (r = {})'.format(r))

time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes) # Position the text

ax.set_xlim([-10, 10])
ax.set_ylim([-1, 10])
ax.set_xlabel('x')
ax.set_ylabel('u(x, t)')
ax.legend(loc='best', ncol=3)

speedup = 2

def update(frame):
    line1.set_ydata(FOM_data[:, frame])
    line2.set_ydata(Galerkin_ROM_data[:, frame])
    line3.set_ydata(OpInf_ROM_data[:, frame])

    time_text.set_text(f"t = {frame * dt:.2f}s") # Update text content
    
    return line1, line2, line3, time_text


all_frames  = np.arange(FOM_data.shape[1])
frames      = all_frames[::speedup]

ani = animation.FuncAnimation(fig, update, frames=frames, blit=True, interval=20)

tight_layout()

show()

ani.save('animations/ROM_solutions.gif', writer='pillow')