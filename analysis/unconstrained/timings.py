
# KUKA
#  DDP      =  0.0012729454804721201  ±  7.184014244231122e-05
#  FDDP     =  0.0012946259530339597  ±  7.911383836089339e-05
#  FDDP_LS  =  0.0012894834670354072  ±  6.153018952347711e-05
#  SQP      =  0.0012895163413882446  ±  6.368322985446061e-05

# PENDULUM
#  DDP      =  0.007536630440878056  ±  0.0023201423141219466
#  FDDP     =  0.009560386866073488  ±  0.0023067396671562256
#  FDDP_LS  =  0.009172463853995954  ±  0.0011853600237480715
#  SQP      =  0.008574847086472666  ±  0.0008257810186892844

# QUADROTOR 
#  DDP      =  0.0005654333411325002  ±  0.00015347977080762748
#  FDDP     =  0.0005132857817721445  ±  6.272549235445462e-05
#  FDDP_LS  =  0.0005001691924962936  ±  2.1469450766082447e-05
#  SQP      =  0.00045154549281226216  ±  1.7919104324827164e-05

# TAICHI
#  DDP      =  0.0763195102805348  ±  0.002204728627072052
#  FDDP     =  0.07692396276051267  ±  0.0024015259138921827
#  FDDP_LS  =  0.07643597870891058  ±  0.002634458753534008
#  SQP      =  0.06881684795170281  ±  0.0017213882554702941


import matplotlib.pyplot as plt
import numpy as np
# kuka, pendulum quadritir, taichi
SOLVERS  = ['DDP', 'FDDP', 'FDDP_LS', 'SQP']
PROBLEMS = ['Kuka', 'Pendulum', 'Quadrotor', 'Taichi']
solving_times = {
    'DDP':     [1e3*0.0012729454804721201, 1e3*0.007536630440878056, 1e3*0.0005654333411325002, 1e3*0.0763195102805348],
    'FDDP':    [1e3*0.0012729454804721201, 1e3*0.009560386866073488, 1e3*0.0005132857817721445, 1e3*0.07692396276051267],
    'FDDP_LS': [1e3*0.0012894834670354072, 1e3*0.009172463853995954, 1e3*0.0005001691924962936, 1e3*0.07643597870891058],
    'SQP':     [1e3*0.0012895163413882446, 1e3*0.008574847086472666, 1e3*0.0004515454928122621, 1e3*0.06881684795170281],
}
colors     = ['r', 'y', 'g', 'b']
linestyles = ['dashdot', 'dashed', 'dotted', 'solid']
linewidths = [4, 4, 4, 4]
labels     = ['DDP', 'FDDP (default LS)', 'FDDP (filter LS)', 'SQP']

# Calculate the number of bars needed
n_solvers = 4
n_problems = 4
total_bars = n_solvers * n_problems

# Define bar width and positions
bar_width = 0.2
index = np.arange(n_problems)

# Plotting the bars
fig, ax = plt.subplots(figsize=(13.8,10.8), constrained_layout=True)

for i in range(n_solvers):
    ax.bar(index + i * bar_width, solving_times[SOLVERS[i]], bar_width, color=colors[i], label=labels[i])

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
save_path = "/tmp/unconstrained_timings.pdf"
print("Saving figure to "+str(save_path))
fig.savefig(save_path, bbox_inches="tight")
plt.show()