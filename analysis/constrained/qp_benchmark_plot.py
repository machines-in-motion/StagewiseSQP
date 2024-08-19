'''
Plot the QP benchmarks
'''
import numpy as np

from plot_config import LABELS, COLORS, LINESTYLES
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42



SAVE_PLOT = True


# Solvers
SOLVERS = ['CSQP',
           'OSQP',
           'HPIPM_DENSE', 
           'HPIPM_OCP']



# name = 'solo12'
# name = 'Kuka'
name = 'Taichi'

if(name == 'Taichi'):
    MAX_QP_TIME = int(1e5)     # in ms
    MAX_QP_ITER = 1000
    TIME_DISCRETIZATION = 1.  # the larger the faster (usefull for very fast problems) 
if(name == 'Kuka'):
    MAX_QP_TIME = int(1e1)       # in ms
    MAX_QP_ITER = 100  
    TIME_DISCRETIZATION = 0.01  # the larger the faster (usefull for very fast problems) 
if(name == 'solo12'):
    MAX_QP_TIME = int(5e3)     # in ms
    MAX_QP_ITER = 100000
    TIME_DISCRETIZATION = 0.1  # the larger the faster (usefull for very fast problems) 




file_name = 'data/' + name + "_qp_benchmark.npz"
print("Loading " + file_name)
npzfile = np.load(file_name)

if('CSQP' in SOLVERS):
    csqp_iter_solved_samples = npzfile["csqp_iter_solved_samples"]
    csqp_time_samples   = npzfile["csqp_time_samples"]
    csqp_iter_samples   = npzfile["csqp_iter_samples"]

if('OSQP' in SOLVERS):
    osqp_iter_solved_samples = npzfile["osqp_iter_solved_samples"]
    osqp_time_samples   = npzfile["osqp_time_samples"]
    osqp_iter_samples   = npzfile["osqp_iter_samples"]

if('HPIPM_DENSE' in SOLVERS):
    hpipm_dense_solved_sample  = npzfile["hpipm_dense_solved_samples"]
    hpipm_dense_time_samples   = npzfile["hpipm_dense_time_samples"]
    hpipm_dense_iter_samples   = npzfile["hpipm_dense_iter_samples"]

if('HPIPM_OCP' in SOLVERS):
    hpipm_ocp_solved_sample  = npzfile["hpipm_ocp_solved_samples"]
    hpipm_ocp_time_samples   = npzfile["hpipm_ocp_time_samples"]
    hpipm_ocp_iter_samples   = npzfile["hpipm_ocp_iter_samples"]

N_samples = len(csqp_iter_solved_samples)
print("N_samples = ", N_samples)


# Compute convergence statistics
TIME_VECTOR_SIZE = int((MAX_QP_TIME + 1) / TIME_DISCRETIZATION)

if('CSQP' in SOLVERS):  
    csqp_iter_solved = np.zeros(MAX_QP_ITER)
    csqp_time_solved = np.zeros(TIME_VECTOR_SIZE)
if('OSQP' in SOLVERS):  
    osqp_iter_solved = np.zeros(MAX_QP_ITER)
    osqp_time_solved = np.zeros(TIME_VECTOR_SIZE)
if('HPIPM_DENSE' in SOLVERS):  
    hpipm_dense_iter_solved = np.zeros(MAX_QP_ITER)
    hpipm_dense_time_solved = np.zeros(TIME_VECTOR_SIZE)
if('HPIPM_OCP' in SOLVERS): 
    hpipm_ocp_iter_solved = np.zeros(MAX_QP_ITER)
    hpipm_ocp_time_solved = np.zeros(TIME_VECTOR_SIZE)

# Count number of problems solved for each sample initial state 
for i in range(N_samples):
    # For sample i of problem k , compare nb iter to max iter
    if('CSQP' in SOLVERS): 
        csqp_iter_ik  = np.array(csqp_iter_samples)[i]
        csqp_time_ik  = np.array(csqp_time_samples)[i]
    if('OSQP' in SOLVERS): 
        osqp_iter_ik = np.array(osqp_iter_samples)[i]
        osqp_time_ik = np.array(osqp_time_samples)[i]
    if('HPIPM_DENSE' in SOLVERS): 
        hpipm_dense_iter_ik = np.array(hpipm_dense_iter_samples)[i]
        hpipm_dense_time_ik = np.array(hpipm_dense_time_samples)[i]
    if('HPIPM_OCP' in SOLVERS): 
        hpipm_ocp_iter_ik = np.array(hpipm_ocp_iter_samples)[i]
        hpipm_ocp_time_ik = np.array(hpipm_ocp_time_samples)[i]
    # Number of iterations
    for j in range(MAX_QP_ITER):
        if('CSQP' in SOLVERS): 
            if(csqp_iter_ik < j): csqp_iter_solved[j] += 1
        if('OSQP' in SOLVERS): 
            if(osqp_iter_ik < j): osqp_iter_solved[j] += 1
        if('HPIPM_DENSE' in SOLVERS): 
            if(hpipm_dense_iter_ik < j): hpipm_dense_iter_solved[j] += 1
        if('HPIPM_OCP' in SOLVERS): 
            if(hpipm_ocp_iter_ik < j): hpipm_ocp_iter_solved[j] += 1
    # Solve time
    for j in range(TIME_VECTOR_SIZE):
        if('CSQP' in SOLVERS): 
            if(csqp_time_ik < j * TIME_DISCRETIZATION): csqp_time_solved[j] += 1
        if('OSQP' in SOLVERS): 
            if(osqp_time_ik < j * TIME_DISCRETIZATION): osqp_time_solved[j] += 1
        if('HPIPM_DENSE' in SOLVERS): 
            if(hpipm_dense_time_ik < j * TIME_DISCRETIZATION): hpipm_dense_time_solved[j] += 1
        if('HPIPM_OCP' in SOLVERS): 
            if(hpipm_ocp_time_ik < j * TIME_DISCRETIZATION): hpipm_ocp_time_solved[j] += 1


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


# x-axis : max number of iterations
xdata_iter = range(0,MAX_QP_ITER)

fig0, (ax1, ax2) = plt.subplots(1, 2, figsize=(19.2,10.8))

# x-axis : max time allowed to solve the QP (in ms)

xdata_time = np.linspace(0, MAX_QP_TIME, TIME_VECTOR_SIZE)
if('CSQP' in SOLVERS): 
    ax1.plot(xdata_time, csqp_time_solved[:]/N_samples, color=COLORS['CSQP'], linestyle=LINESTYLES['CSQP'], linewidth=4, label=LABELS['CSQP']) 
if('OSQP' in SOLVERS): 
    ax1.plot(xdata_time, osqp_time_solved[:]/N_samples, color=COLORS['OSQP'], linestyle=LINESTYLES['OSQP'], linewidth=4, label=LABELS['OSQP'])
if('HPIPM_DENSE' in SOLVERS): 
    ax1.plot(xdata_time, hpipm_dense_time_solved[:]/N_samples, color=COLORS['HPIPM_DENSE'], linestyle=LINESTYLES['HPIPM_DENSE'], linewidth=4, label=LABELS['HPIPM_DENSE'])
if('HPIPM_OCP' in SOLVERS): 
    ax1.plot(xdata_time, hpipm_ocp_time_solved[:]/N_samples, color=COLORS['HPIPM_OCP'], linestyle=LINESTYLES['HPIPM_OCP'], linewidth=4, label=LABELS['HPIPM_OCP'])

# Set axis and stuff
ax1.set_ylabel('Percentage of problems solved', fontsize=26)
ax1.set_xlabel('Max. solving time (ms)', fontsize=26)
ax1.set_ylim(-0.02, 1.02)
ax1.set_xscale("log")
ax1.tick_params(axis = 'y', labelsize=22)
ax1.tick_params(axis = 'x', labelsize=22)
ax1.set_xscale('log')
ax1.grid(True) 


if('CSQP' in SOLVERS): 
    ax2.plot(xdata_iter, csqp_iter_solved[:]/N_samples, color=COLORS['CSQP'], linestyle=LINESTYLES['CSQP'], linewidth=4, label=LABELS['CSQP']) 
if('OSQP' in SOLVERS): 
    ax2.plot(xdata_iter, osqp_iter_solved[:]/N_samples, color=COLORS['OSQP'], linestyle=LINESTYLES['OSQP'], linewidth=4, label=LABELS['OSQP'])
if('HPIPM_DENSE' in SOLVERS): 
    ax2.plot(xdata_iter, hpipm_dense_iter_solved[:]/N_samples, color=COLORS['HPIPM_DENSE'], linestyle=LINESTYLES['HPIPM_DENSE'], linewidth=4, label=LABELS['HPIPM_DENSE'])
if('HPIPM_OCP' in SOLVERS): 
    ax2.plot(xdata_iter, hpipm_ocp_iter_solved[:]/N_samples, color=COLORS['HPIPM_OCP'], linestyle=LINESTYLES['HPIPM_OCP'], linewidth=4, label=LABELS['HPIPM_OCP'])
# Set axis and stuff
ax2.set_xlabel('Max. number of iterations', fontsize=26)
ax2.set_ylim(-0.02, 1.02)
ax2.set_xscale('log')
ax2.tick_params(axis = 'y', labelsize=22)
ax2.tick_params(axis = 'x', labelsize=22)
ax2.grid(True) 
# Legend 
handles0, labels0 = ax2.get_legend_handles_labels()
fig0.legend(handles0, labels0, loc='upper right', prop={'size': 26}) 
plt.gca().axes.yaxis.set_ticklabels([])
# plt.tight_layout()

if(SAVE_PLOT):
    fig0.savefig('figures/qp_bench_' + name + '.pdf', bbox_inches="tight")
    fig0.savefig('/home/skleff/SQP_REBUTAL_BENCH/constrained/qp_bench_' + name + '.pdf', bbox_inches="tight")

plt.show()
plt.close('all')