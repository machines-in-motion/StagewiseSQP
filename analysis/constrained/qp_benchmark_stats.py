import numpy as np

from plot_config import LABELS, COLORS, LINESTYLES
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

SAVE_PLOT = True

# Solvers
SOLVERS = ['CSQP', 'OSQP', 'HPIPM_DENSE', 'HPIPM_OCP']
PROBLEMS = ['solo12', 'Kuka', 'Taichi']

# Function to compute statistics and format them into LaTeX with debug prints
def compute_and_format_statistics(name, times):
    if len(times) == 0:
        print(f"Warning: No data available for {name}.")
        return "N/A & N/A & N/A \\\\ \hline\n"

    median_time = np.median(times)
    q1_time = np.percentile(times, 25)
    q3_time = np.percentile(times, 75)
    
    # Debug prints
    print(f"  Median: {median_time:.4f}")
    print(f"  1st Quartile (25th percentile): {q1_time:.4f}")
    print(f"  3rd Quartile (75th percentile): {q3_time:.4f}")
    
    return f"{median_time:.4f} & {q1_time:.4f} & {q3_time:.4f} \\\\ \hline\n"

# Initialize LaTeX table content and printout content
printout_lines = []

PREFIX = "data/"
for problem in PROBLEMS:
    npz_file = np.load(PREFIX + problem + "_qp_benchmark.npz")
    
    printout_lines.append(f"\nStatistics for problem: {problem}")
    printout_lines.append("-" * 80)
    
    for solver in SOLVERS:
        time_samples = npz_file.get(solver.lower() + '_time_samples', [])
        iter_samples = npz_file.get(solver.lower() + '_iter_samples', [])
        
        if len(time_samples) == 0:
            printout_lines.append(f"Solver: {solver}")
            printout_lines.append("  No data available.")
            printout_lines.append("-" * 80)
            continue
        if len(iter_samples) == 0:
            printout_lines.append(f"Solver: {solver}")
            printout_lines.append("  No data available.")
            printout_lines.append("-" * 80)
            continue

        
        # Compute statistics
        median_time = np.median(time_samples)
        q1_time = np.percentile(time_samples, 25)
        q3_time = np.percentile(time_samples, 75)

        median_iter = np.median(iter_samples)
        q1_iter = np.percentile(iter_samples, 25)
        q3_iter = np.percentile(iter_samples, 75)

        # Print to console
        printout_lines.append(f"Solver: {solver}")
        printout_lines.append(" TIME :")
        # printout_lines.append(f"  1st Quartile: {q1_time:.4f}")
        printout_lines.append(f"  Median time: {median_time:.4f}")
        # printout_lines.append(f"  3rd Quartile: {q3_time:.4f}")
        printout_lines.append(" ITER :")
        # printout_lines.append(f"  1st Quartile: {q1_iter:.4f}")
        printout_lines.append(f"  Median iter: {median_iter:.4f}")
        # printout_lines.append(f"  3rd Quartile: {q3_iter:.4f}")
        printout_lines.append("-" * 80)

# Print statistics to console
print("\n".join(printout_lines))
