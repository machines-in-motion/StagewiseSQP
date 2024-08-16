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

# Solvers
SOLVERS = ['DDP',
           'FDDP',
           'FDDP_filter', 
           'SQP']

PROBLEMS = ['Kuka', 
            'Quadrotor', 
            'Pendulum', 
            'Taichi']

npz_data = {}

# Load data and display timings
if('Kuka' in PROBLEMS):
    kuka_name      = "data/Kuka.npz"
    print("Loading " + kuka_name)
    npz_kuka = np.load(kuka_name)
    npz_data['Kuka'] = npz_kuka
    N_SAMPLES_kuka = npz_kuka['N_SAMPLES']
    MAXITER_kuka   = npz_kuka['MAXITER']
    print("Average solving time per iteration Kuka \n")
    print(" DDP          = " , npz_kuka['ddp_mean_solve_time']         ,  ' \xB1 ' , npz_kuka['ddp_std_solve_time'])
    print(" FDDP         = " , npz_kuka['fddp_mean_solve_time']        ,  ' \xB1 ' , npz_kuka['fddp_std_solve_time'])
    print(" FDDP_filter  = " , npz_kuka['fddp_filter_mean_solve_time'] ,  ' \xB1 ' , npz_kuka['fddp_filter_std_solve_time'])
    print(" SQP          = " , npz_kuka['SQP_mean_solve_time']         ,  ' \xB1 ' , npz_kuka['SQP_std_solve_time'])

if('Quadrotor' in PROBLEMS):
    quadrotor_name = "data/Quadrotor.npz"
    print("Loading " + quadrotor_name)
    npz_quadrotor = np.load(quadrotor_name)
    npz_data['Quadrotor'] = npz_quadrotor
    N_SAMPLES_quadrotor = npz_quadrotor['N_SAMPLES']
    MAXITER_quadrotor   = npz_quadrotor['MAXITER']
    print("Average solving time per iteration Quadrotor \n")
    print(" DDP          = " , npz_quadrotor['ddp_mean_solve_time']         ,  ' \xB1 ' , npz_quadrotor['ddp_std_solve_time'])
    print(" FDDP         = " , npz_quadrotor['fddp_mean_solve_time']        ,  ' \xB1 ' , npz_quadrotor['fddp_std_solve_time'])
    print(" FDDP_filter  = " , npz_quadrotor['fddp_filter_mean_solve_time'] ,  ' \xB1 ' , npz_quadrotor['fddp_filter_std_solve_time'])
    print(" SQP          = " , npz_quadrotor['SQP_mean_solve_time']         ,  ' \xB1 ' , npz_quadrotor['SQP_std_solve_time'])

if('Pendulum' in PROBLEMS):
    pendulum_name  = "data/Pendulum.npz"
    print("Loading " + pendulum_name)
    npz_pendulum = np.load(pendulum_name)
    npz_data['Pendulum'] = npz_pendulum
    N_SAMPLES_pendulum = npz_pendulum['N_SAMPLES']
    MAXITER_pendulum   = npz_pendulum['MAXITER']
    print("Average solving time per iteration Pendulum \n")
    print(" DDP          = " , npz_pendulum['ddp_mean_solve_time']         ,  ' \xB1 ' , npz_pendulum['ddp_std_solve_time'])
    print(" FDDP         = " , npz_pendulum['fddp_mean_solve_time']        ,  ' \xB1 ' , npz_pendulum['fddp_std_solve_time'])
    print(" FDDP_filter  = " , npz_pendulum['fddp_filter_mean_solve_time'] ,  ' \xB1 ' , npz_pendulum['fddp_filter_std_solve_time'])
    print(" SQP          = " , npz_pendulum['SQP_mean_solve_time']         ,  ' \xB1 ' , npz_pendulum['SQP_std_solve_time'])

if('Taichi' in PROBLEMS):
    taichi_name    = "data/Taichi.npz"
    print("Loading " + taichi_name)
    npz_taichi = np.load(taichi_name)
    npz_data['Taichi'] = npz_taichi
    N_SAMPLES_taichi = npz_taichi['N_SAMPLES']
    MAXITER_taichi   = npz_taichi['MAXITER']
    print("Average solving time per iteration Taichi \n")
    print(" DDP          = " , npz_taichi['ddp_mean_solve_time']         ,  ' \xB1 ' , npz_taichi['ddp_std_solve_time'])
    print(" FDDP         = " , npz_taichi['fddp_mean_solve_time']        ,  ' \xB1 ' , npz_taichi['fddp_std_solve_time'])
    print(" FDDP_filter  = " , npz_taichi['fddp_filter_mean_solve_time'] ,  ' \xB1 ' , npz_taichi['fddp_filter_std_solve_time'])
    print(" SQP          = " , npz_taichi['SQP_mean_solve_time']         ,  ' \xB1 ' , npz_taichi['SQP_std_solve_time'])

solving_times = {
    'DDP':         [1e3*npz_kuka['ddp_mean_solve_time'] , 1e3*npz_quadrotor['ddp_mean_solve_time'], 1e3*npz_pendulum['ddp_mean_solve_time'] , 1e3*npz_taichi['ddp_mean_solve_time'] ],
    'FDDP':        [1e3*npz_kuka['fddp_mean_solve_time'] , 1e3*npz_quadrotor['fddp_mean_solve_time'], 1e3*npz_pendulum['fddp_mean_solve_time'] , 1e3*npz_taichi['fddp_mean_solve_time'] ],
    'FDDP_filter': [1e3*npz_kuka['fddp_filter_mean_solve_time'] , 1e3*npz_quadrotor['fddp_filter_mean_solve_time'], 1e3*npz_pendulum['fddp_filter_mean_solve_time'] , 1e3*npz_taichi['fddp_filter_mean_solve_time'] ],
    'SQP':         [1e3*npz_kuka['SQP_mean_solve_time'] , 1e3*npz_quadrotor['SQP_mean_solve_time'], 1e3*npz_pendulum['SQP_mean_solve_time'] , 1e3*npz_taichi['SQP_mean_solve_time'] ],
}

# Calculate the number of bars needed
n_solvers = len(SOLVERS)
n_problems = len(PROBLEMS)
total_bars = n_solvers * n_problems

# Define bar width and positions
bar_width = 0.2
index = np.arange(n_problems)

# Plotting the bars
fig, ax = plt.subplots(figsize=(13.8,10.8), constrained_layout=True)

for i, solver in enumerate(SOLVERS):
    print(solver)
    ax.bar(index + i * bar_width, solving_times[solver], bar_width, color=COLORS[solver], label=LABELS[solver])

# Adding labels, title, and legend
ax.set_xlabel('Problems', fontsize=26)
ax.grid(True)
ax.set_yscale('log')
ax.set_ylabel('Average time per iteration (ms)', fontsize=26)
ax.set_title('Average time per iteration', fontsize=26)
ax.set_xticks(index + bar_width * (n_solvers - 1) / 2)
ax.set_xticklabels(PROBLEMS)
ax.tick_params(axis = 'y', labelsize=22)
ax.tick_params(axis = 'x', labelsize=22)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, 0.95), prop={'size': 26}) 
# Save, show , clean
if(SAVE_PLOT):
    fig.savefig('/home/skleff/SQP_REBUTAL_BENCH/rollout_timings.pdf', bbox_inches="tight")
    fig.savefig('figures/rollout_timings.pdf', bbox_inches="tight")
plt.show()