import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from plot_config import LABELS, COLORS, LINESTYLES, LABELSIZE, FONTSIZE, FIGSIZE_SQ

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SAVE_PLOT = False

# Solvers
SOLVERS = ['DDP', 'FDDP', 'FDDP_filter', 'SQP']
PROBLEMS = ['Kuka', 'Quadrotor', 'Pendulum', 'Taichi']

npz_data = {}

# Load data and display timings
PREFIX = "data/"
data_files = {problem: PREFIX + f"{problem}.npz" for problem in PROBLEMS}

for problem, filename in data_files.items():
    print(f"Loading {filename}")
    npz_file = np.load(filename)
    npz_data[problem] = npz_file

    print(f"Average solving time per iteration {problem} \n")
    for solver in SOLVERS:
        mean_time = npz_file[f'{solver.lower()}_mean_solve_time']
        std_time = npz_file[f'{solver.lower()}_std_solve_time']
        print(f" {solver: <12} = {1e3*mean_time:.4f} \xB1 {1e3*std_time:.4f}")

# Collect solving times and variances
solving_times = {solver: [] for solver in SOLVERS}
variances = {solver: [] for solver in SOLVERS}

for problem in PROBLEMS:
    npz_file = npz_data[problem]
    for solver in SOLVERS:
        mean_time = npz_file[f'{solver.lower()}_mean_solve_time']
        std_time = npz_file[f'{solver.lower()}_std_solve_time']
        solving_times[solver].append(1e3 * mean_time)
        variances[solver].append(1e3 * std_time)

# Calculate the number of bars needed
n_solvers = len(SOLVERS)
n_problems = len(PROBLEMS)
total_bars = n_solvers * n_problems

# Define bar width and positions
bar_width = 0.2
index = np.arange(n_problems)

# Plotting the bars with error bars
fig, ax = plt.subplots(figsize=FIGSIZE_SQ, constrained_layout=True)

for i, solver in enumerate(SOLVERS):
    ax.bar(index + i * bar_width, solving_times[solver], bar_width, yerr=variances[solver],
           color=COLORS[solver], label=LABELS[solver], capsize=5, alpha=0.8, edgecolor='black')

# Adding labels, title, and legend
ax.set_xlabel('Problems', fontsize=FONTSIZE)
ax.grid(True)
ax.set_yscale('log')
ax.set_ylabel('Average time per iteration (ms)', fontsize=FONTSIZE)
ax.set_title('Average time per iteration', fontsize=FONTSIZE)
ax.set_xticks(index + bar_width * (n_solvers - 1) / 2)
ax.set_xticklabels(PROBLEMS)
ax.tick_params(axis='y', labelsize=LABELSIZE)
ax.tick_params(axis='x', labelsize=LABELSIZE)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.1, 0.95), prop={'size': FONTSIZE})

# Save, show, clean
if SAVE_PLOT:
    fig.savefig('figures/rollout_timings.pdf', bbox_inches="tight")
    fig.savefig('/home/skleff/SQP_REBUTAL_BENCH/rollout_timings.pdf', bbox_inches="tight")

plt.show()