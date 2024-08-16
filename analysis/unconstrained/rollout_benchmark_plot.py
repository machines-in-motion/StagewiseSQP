'''
Plot the rollout benchmarks
'''
import numpy as np

from plot_config import LABELS, COLORS, LINESTYLES, LABELSIZE, FONTSIZE, FIGSIZE
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SAVE_PLOT = True

# Benchmark name 
BENCH_NAME = 'Taichi'

# Solvers
SOLVERS = ['DDP',
           'FDDP',
           'FDDP_filter', 
           'SQP']
# Load data 
file_name = "/home/skleff/SQP_REBUTAL_BENCH/"+BENCH_NAME + ".npz"
print("Loading " + file_name)
npzfile = np.load(file_name)
N_SAMPLES = npzfile['N_SAMPLES']
MAXITER   = npzfile['MAXITER']
print("N_SAMPLES = ", N_SAMPLES)
print("MAXITER   = ", MAXITER)


# x-axis : max number of iterations
xdata     = range(0,MAXITER)
xdata2     = range(0,N_SAMPLES)
# Plot number of problem solved vs max number of iterations
fig0, ax0 = plt.subplots(1, 1, figsize=FIGSIZE)
ax0.plot(xdata, npzfile['ddp_iter_solved']/N_SAMPLES, color=COLORS['DDP'], linewidth=4, linestyle=LINESTYLES['DDP'], label=LABELS['DDP']) 
ax0.plot(xdata, npzfile['fddp_iter_solved']/N_SAMPLES, color=COLORS['FDDP'], linewidth=4, linestyle=LINESTYLES['FDDP'], label=LABELS['FDDP']) 
ax0.plot(xdata, npzfile['fddp_filter_iter_solved']/N_SAMPLES, color=COLORS['FDDP_filter'], linewidth=4, linestyle=LINESTYLES['FDDP_filter'], label=LABELS['FDDP_filter']) 
ax0.plot(xdata, npzfile['SQP_iter_solved']/N_SAMPLES, color=COLORS['SQP'], linewidth=4, linestyle=LINESTYLES['SQP'], label=LABELS['SQP']) 
# Set axis and stuff
ax0.set_ylabel('Percentage of problems solved', fontsize=FONTSIZE)
ax0.set_xlabel('Max. number of iterations', fontsize=FONTSIZE)
ax0.set_ylim(-0.02, 1.02)
ax0.tick_params(axis = 'y', labelsize=LABELSIZE)
ax0.tick_params(axis = 'x', labelsize=LABELSIZE)
ax0.grid(True) 
# Legend 
handles0, labels0 = ax0.get_legend_handles_labels()
fig0.legend(handles0, labels0, loc='lower right', bbox_to_anchor=(0.902, 0.1), prop={'size': FONTSIZE}) 
# Save, show , clean
if(SAVE_PLOT):
    fig0.savefig('/home/skleff/SQP_REBUTAL_BENCH/bench_'+BENCH_NAME+'.pdf', bbox_inches="tight")

plt.show()
plt.close('all')