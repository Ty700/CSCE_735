import matplotlib.pyplot as plt
import numpy as np

######## Trials 10^ 8 ########
######## Plotting Execution Time vs Number of Threads ########

# Data from the table
threads = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
time = [0.9385, 0.4921, 0.2449, 0.1262, 0.0639, 0.033, 0.0312, 0.0257, 0.0275, 0.0297, 0.0331, 0.0659, 0.1328, 0.2713]

# Create the plot
plt.figure(figsize=(10, 6))
plt.semilogx(threads, time, marker='o', linewidth=2, markersize=8)

# Labels and title
plt.xlabel('Number of Threads (p)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Execution Time vs Number of Threads (n=10^8 trials)', fontsize=14, fontweight='bold')

# Grid for better readability
plt.grid(True, which='both', linestyle='--', alpha=0.6)

# Set x-axis ticks to match the data points
plt.xticks(threads, [str(t) for t in threads], rotation=45)

plt.tight_layout()
plt.savefig('execution_time_vs_threads.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Minimum execution time: {min(time):.4f} seconds at {threads[time.index(min(time))]} threads")
print(f"Maximum execution time: {max(time):.4f} seconds at {threads[time.index(max(time))]} threads")

######## Calculating and Printing Speedup ########

threads = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
time = [0.9385, 0.4921, 0.2449, 0.1262, 0.0639, 0.033, 0.0312, 0.0257, 0.0275, 0.0297, 0.0331, 0.0659, 0.1328, 0.2713]

T_sequential = time[0]  
speedup = [T_sequential / t for t in time]

for p, s in zip(threads, speedup):
    print(f"Threads: {p:4d}, Speedup: {s:6.2f}x")

######## Plotting Speedup vs Number of Threads ########
# Create the speedup plot
plt.figure(figsize=(10, 6))
plt.semilogx(threads, speedup, marker='o', color='orange', linewidth=2, markersize=8)
# Labels and title
plt.xlabel('Number of Threads (p)', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('Speedup vs Number of Threads (n=10^8 trials)', fontsize=14, fontweight='bold')
# Grid for better readability
plt.grid(True, which='both', linestyle='--', alpha=0.6)
# Set x-axis ticks to match the data points
plt.xticks(threads, [str(t) for t in threads], rotation=45)
plt.tight_layout()
plt.savefig('speedup_vs_threads.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"Maximum speedup: {max(speedup):.2f}x at {threads[speedup.index(max(speedup))]} threads")

######### Plotting Efficiency vs Number of Threads ########
efficiency = [s / p for s, p in zip(speedup, threads)]
# Create the efficiency graph 
plt.figure(figsize=(10, 6))
plt.semilogx(threads, efficiency, marker='o', color='green', linewidth=2, markersize=8)
# Labels and title
plt.xlabel('Number of Threads (p)', fontsize=12)
plt.ylabel('Efficiency', fontsize=12)
plt.title('Efficiency vs Number of Threads (n=10^8 trials)', fontsize=14, fontweight='bold')
# Grid for better readability
plt.grid(True, which='both', linestyle='--', alpha=0.6)
# Set x-axis ticks to match the data points
plt.xticks(threads, [str(t) for t in threads], rotation=45)
plt.tight_layout()
plt.savefig('efficiency_vs_threads.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"Maximum efficiency: {max(efficiency):.4f} at {threads[efficiency.index(max(efficiency))]} threads")

####### Trials 10 ^ 10 ########
######## Plotting Execution Time vs Number of Threads ########
# Data from the table
threads_10e10 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
time_10e10 = [93.5356, 48.0348, 24.5289, 12.5004, 6.315, 3.1565, 2.4943, 2.2656, 2.2046, 2.2202, 2.2052, 2.285, 2.2371, 2.3622]

# Create the plot 
plt.figure(figsize=(10, 6))
plt.semilogx(threads_10e10, time_10e10, marker='o', linewidth=2, markersize=8)
# Labels and title
plt.xlabel('Number of Threads (p)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Execution Time vs Number of Threads (n=10^10 trials)', fontsize=14, fontweight='bold')
# Grid for better readability
plt.grid(True, which='both', linestyle='--', alpha=0.6)
# Set x-axis ticks to match the data points
plt.xticks(threads_10e10, [str(t) for t in threads_10e10], rotation=45)
plt.tight_layout()
plt.savefig('execution_time_vs_threads_2_1.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"Minimum execution time: {min(time_10e10):.4f} seconds at {threads_10e10[time_10e10.index(min(time_10e10))]} threads") 

######## Plotting Error vs Number of Trials ########

# Data from experiments (n = 10^k for k=3 to 9, p=48)
n_values = [1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000]
errors = [4.76e-02, 1.01e-02, 4.29e-03, 1.21e-03, 1.29e-03, 2.08e-04, 8.04e-06]

# Create the plot
plt.figure(figsize=(10, 6))
plt.loglog(n_values, errors, marker='o', linewidth=2, markersize=8, 
           label='Experimental Error', color='blue')

# Add theoretical 1/sqrt(n) line for comparison
# Error should be proportional to 1/sqrt(n) for Monte Carlo
# Fit to first point to get constant
C = errors[0] * np.sqrt(n_values[0])
theoretical_error = C / np.sqrt(np.array(n_values))
plt.loglog(n_values, theoretical_error, '--', linewidth=2, 
           label=r'Theoretical $\propto 1/\sqrt{n}$', color='red', alpha=0.7)

# Labels and title
plt.xlabel('Number of Trials (n)', fontsize=12)
plt.ylabel('Absolute Error', fontsize=12)
plt.title('Error vs Number of Trials (p=48 threads)', fontsize=14, fontweight='bold')

# Grid for better readability
plt.grid(True, which='both', linestyle='--', alpha=0.6)

# Legend
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('error_vs_trials.png', dpi=300, bbox_inches='tight')

plt.show()

print("Plot saved successfully!")
print(f"\nError Analysis:")
print(f"n = 10^3:  error = {errors[0]:.2e}")
print(f"n = 10^9:  error = {errors[-1]:.2e}")
print(f"Error reduction factor: {errors[0]/errors[-1]:.1f}x")
print(f"Theoretical reduction (sqrt(10^6)): {np.sqrt(1e6):.1f}x")

######## Part 2: MPI with 10^10 trials ########

# Data from MPI experiments
processes = [1, 4, 8, 16, 32, 64]
time_mpi = [17.2724, 4.4837, 2.2923, 1.1349, 1.0593, 0.2899]

######## Question 5.1: Execution Time vs Number of Processes ########
plt.figure(figsize=(10, 6))
plt.semilogx(processes, time_mpi, marker='o', linewidth=2, markersize=8, color='blue')

plt.xlabel('Number of Processes (p)', fontsize=12)
plt.ylabel('Execution Time (seconds)', fontsize=12)
plt.title('Execution Time vs Number of Processes (n=10^10 trials, MPI)', fontsize=14, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xticks(processes, [str(p) for p in processes])
plt.tight_layout()
plt.savefig('execution_time_vs_processes_5_1.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Minimum execution time: {min(time_mpi):.4f} seconds at {processes[time_mpi.index(min(time_mpi))]} processes")
print()

######## Question 5.2: Speedup vs Number of Processes ########
T_sequential = time_mpi[0]
speedup_mpi = [T_sequential / t for t in time_mpi]

for p, s in zip(processes, speedup_mpi):
    print(f"Processes: {p:2d}, Speedup: {s:6.2f}x")
print()

plt.figure(figsize=(10, 6))
plt.semilogx(processes, speedup_mpi, marker='o', color='orange', linewidth=2, markersize=8)

# Add ideal linear speedup line for comparison
plt.semilogx(processes, processes, '--', color='gray', linewidth=2, alpha=0.7, label='Ideal Linear Speedup')

plt.xlabel('Number of Processes (p)', fontsize=12)
plt.ylabel('Speedup', fontsize=12)
plt.title('Speedup vs Number of Processes (n=10^10 trials, MPI)', fontsize=14, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xticks(processes, [str(p) for p in processes])
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig('speedup_vs_processes_5_2.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Maximum speedup: {max(speedup_mpi):.2f}x at {processes[speedup_mpi.index(max(speedup_mpi))]} processes")
print()

######## Question 5.3: Efficiency vs Number of Processes ########
efficiency_mpi = [s / p for s, p in zip(speedup_mpi, processes)]

for p, e in zip(processes, efficiency_mpi):
    print(f"Processes: {p:2d}, Efficiency: {e:6.4f} ({e*100:.2f}%)")
print()

plt.figure(figsize=(10, 6))
plt.semilogx(processes, efficiency_mpi, marker='o', color='green', linewidth=2, markersize=8)

plt.xlabel('Number of Processes (p)', fontsize=12)
plt.ylabel('Efficiency', fontsize=12)
plt.title('Efficiency vs Number of Processes (n=10^10 trials, MPI)', fontsize=14, fontweight='bold')
plt.grid(True, which='both', linestyle='--', alpha=0.6)
plt.xticks(processes, [str(p) for p in processes])
plt.tight_layout()
plt.savefig('efficiency_vs_processes_5_3.png', dpi=300, bbox_inches='tight')
plt.show()

############# N-TasksPerNode vs Time (MPI) #############
# Data from the experiment
ntasks_per_node = [1, 2, 4, 8, 16, 32]
time_sec = [1.2634, 0.6926, 0.3001, 0.3135, 0.3157, 0.2905]

# Find the minimum time and corresponding ntasks-per-node
min_time = min(time_sec)
min_ntasks = ntasks_per_node[time_sec.index(min_time)]

print(f"Data Analysis:")
print(f"{'ntasks-per-node':<20} {'Time (sec)':<15}")
print("-" * 35)
for n, t in zip(ntasks_per_node, time_sec):
    marker = " <-- MINIMUM" if t == min_time else ""
    print(f"{n:<20} {t:<15.4f}{marker}")

print(f"\nMinimum time: {min_time:.4f} seconds")
print(f"Optimal ntasks-per-node: {min_ntasks}")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(ntasks_per_node, time_sec, 'b-o', linewidth=2, markersize=8, label='Execution Time')
plt.plot(min_ntasks, min_time, 'r*', markersize=20, label=f'Minimum (ntasks={min_ntasks})')

plt.xlabel('ntasks-per-node', fontsize=12, fontweight='bold')
plt.ylabel('Time (seconds)', fontsize=12, fontweight='bold')
plt.title('Execution Time vs. ntasks-per-node\n(n=10^10, p=64)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.xscale('log', base=2)  # Log scale since values are powers of 2
plt.xticks(ntasks_per_node, ntasks_per_node)

# Add annotation for minimum
plt.annotate(f'Min: {min_time:.4f}s', 
             xy=(min_ntasks, min_time), 
             xytext=(min_ntasks/2, min_time + 0.15),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=11, 
             color='red',
             fontweight='bold')

plt.tight_layout()
plt.savefig('ntasks_analysis_6.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to ntasks_analysis.png")

# Additional analysis
print("\n" + "="*50)
print("Performance Analysis:")
print("="*50)
speedup_from_1 = time_sec[0] / min_time
print(f"Speedup from ntasks=1 to ntasks={min_ntasks}: {speedup_from_1:.2f}x")
print(f"Time reduction: {(1 - min_time/time_sec[0])*100:.1f}%")

# Data extracted from the image
# n values
n_values = [100, 10000, 1000000, 100000000]  # 10^2, 10^4, 10^6, 10^8
n_labels = ['10²', '10⁴', '10⁶', '10⁸']

# p=64 data
time_p64 = [0.0420, 0.0307, 0.0327, 0.0333]
rel_error_p64 = [2.65e-06, 2.65e-10, 2.63e-14, 0.00e+00]

# p=1 data
time_p1 = [0.0000, 0.0000, 0.0018, 0.1763]
rel_error_p1 = [2.65e-06, 2.65e-10, 4.64e-14, 7.29e-14]

print("="*60)
print("Data Summary:")
print("="*60)
print(f"{'n':<15} {'Time p=1':<15} {'Time p=64':<15} {'Speedup':<15}")
print("-"*60)

speedup = []
for i, n in enumerate(n_values):
    if time_p1[i] > 0:
        sp = time_p1[i] / time_p64[i]
        speedup.append(sp)
        print(f"{n_labels[i]:<15} {time_p1[i]:<15.4f} {time_p64[i]:<15.4f} {sp:<15.2f}")
    else:
        speedup.append(None)
        print(f"{n_labels[i]:<15} {time_p1[i]:<15.4f} {time_p64[i]:<15.4f} {'N/A':<15}")

print("\n" + "="*60)
print("Relative Error Summary:")
print("="*60)
print(f"{'n':<15} {'Rel Error (p=1)':<20} {'Rel Error (p=64)':<20}")
print("-"*60)
for i, n in enumerate(n_values):
    print(f"{n_labels[i]:<15} {rel_error_p1[i]:<20.2e} {rel_error_p64[i]:<20.2e}")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Speedup vs n
valid_indices = [i for i, sp in enumerate(speedup) if sp is not None]
valid_n = [n_values[i] for i in valid_indices]
valid_speedup = [speedup[i] for i in valid_indices]
valid_n_labels = [n_labels[i] for i in valid_indices]

ax1.plot(valid_n, valid_speedup, 'b-o', linewidth=2, markersize=10, label='Speedup (p=64 vs p=1)')
ax1.set_xlabel('n (problem size)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Speedup', fontsize=12, fontweight='bold')
ax1.set_title('Speedup vs. Problem Size (n)\np=64 w.r.t. p=1, ntasks-per-node=4', fontsize=13, fontweight='bold')
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Set x-axis ticks to show actual n values
ax1.set_xticks(valid_n)
ax1.set_xticklabels(valid_n_labels)

# Add data labels
for i, (x, y) in enumerate(zip(valid_n, valid_speedup)):
    ax1.annotate(f'{y:.2f}x', xy=(x, y), xytext=(5, 5), 
                textcoords='offset points', fontsize=9, fontweight='bold')

# Add note about missing data
if len(valid_indices) < len(n_values):
    ax1.text(0.05, 0.95, 'Note: Small n values excluded\n(execution time too small to measure)',
            transform=ax1.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot 2: Relative Error vs n (using p=64 data)
# Replace 0.00e+00 with a very small number for log scale plotting
rel_error_plot = [e if e > 0 else 1e-16 for e in rel_error_p64]

ax2.semilogy(n_values, rel_error_plot, 'r-s', linewidth=2, markersize=10, label='Relative Error (p=64)')
ax2.set_xlabel('n (problem size)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Relative Error (log scale)', fontsize=12, fontweight='bold')
ax2.set_title('Relative Error vs. Problem Size (n)\np=64, ntasks-per-node=4', fontsize=13, fontweight='bold')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(fontsize=10)

# Set x-axis ticks
ax2.set_xticks(n_values)
ax2.set_xticklabels(n_labels)

# Add data labels
for i, (x, y_original) in enumerate(zip(n_values, rel_error_p64)):
    if y_original > 0:
        ax2.annotate(f'{y_original:.2e}', xy=(x, rel_error_plot[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8)
    else:
        ax2.annotate('~0', xy=(x, rel_error_plot[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/speedup_and_error_analysis.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to speedup_and_error_analysis.png")

# Additional analysis
print("\n" + "="*60)
print("Analysis:")
print("="*60)
if valid_speedup:
    print(f"Maximum speedup observed: {max(valid_speedup):.2f}x at n={valid_n_labels[valid_speedup.index(max(valid_speedup))]}")
    print(f"Minimum speedup observed: {min(valid_speedup):.2f}x at n={valid_n_labels[valid_speedup.index(min(valid_speedup))]}")
print(f"\nAccuracy improves with larger n:")
print(f"  - Relative error decreases from {rel_error_p64[0]:.2e} (n=10²) to {rel_error_p64[-1]:.2e} (n=10⁸)")

# Data extracted from the image
n_values = [100, 10000, 1000000, 100000000]  # 10^2, 10^4, 10^6, 10^8
n_labels = ['10²', '10⁴', '10⁶', '10⁸']

# p=64 data
time_p64 = [0.0420, 0.0307, 0.0327, 0.0333]
rel_error_p64 = [2.65e-06, 2.65e-10, 2.63e-14, 0.00e+00]

# p=1 data
time_p1 = [0.0000, 0.0000, 0.0018, 0.1763]

# Calculate speedup
speedup = []
for i in range(len(n_values)):
    if time_p1[i] > 0:
        speedup.append(time_p1[i] / time_p64[i])
    else:
        speedup.append(None)

# Question 7.1: Speedup Plot
fig1, ax1 = plt.subplots(figsize=(10, 6))

valid_indices = [i for i, sp in enumerate(speedup) if sp is not None]
valid_n = [n_values[i] for i in valid_indices]
valid_speedup = [speedup[i] for i in valid_indices]
valid_n_labels = [n_labels[i] for i in valid_indices]

ax1.plot(valid_n, valid_speedup, 'b-o', linewidth=2.5, markersize=12, label='Speedup (p=64 vs p=1)')
ax1.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Speedup = 1 (no benefit)')
ax1.set_xlabel('n (Problem Size)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Speedup', fontsize=14, fontweight='bold')
ax1.set_title('Question 7.1: Speedup vs. Problem Size (n)\np=64 w.r.t. p=1, ntasks-per-node=4', 
              fontsize=15, fontweight='bold', pad=20)
ax1.set_xscale('log')
ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
ax1.legend(fontsize=11, loc='best')

# Set x-axis ticks
ax1.set_xticks(valid_n)
ax1.set_xticklabels(valid_n_labels, fontsize=12)
ax1.tick_params(axis='y', labelsize=12)

# Add data labels
for i, (x, y) in enumerate(zip(valid_n, valid_speedup)):
    ax1.annotate(f'{y:.2f}x', xy=(x, y), xytext=(0, 10), 
                textcoords='offset points', fontsize=11, fontweight='bold',
                ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Add note about missing data
if len(valid_indices) < len(n_values):
    note_text = 'Note: n=10² and n=10⁴ excluded\n(p=1 execution time < 0.0001s,\ntoo small for accurate measurement)'
    ax1.text(0.05, 0.95, note_text,
            transform=ax1.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('question_7_1_speedup.png', dpi=300, bbox_inches='tight')
print("Question 7.1 plot saved: question_7_1_speedup.png")

# Question 7.2: Relative Error Plot
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Replace 0.00e+00 with a very small number for log scale plotting
rel_error_plot = [e if e > 0 else 1e-16 for e in rel_error_p64]

ax2.semilogy(n_values, rel_error_plot, 'r-s', linewidth=2.5, markersize=12, 
            label='Relative Error (p=64)', markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2)
ax2.set_xlabel('n (Problem Size)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Relative Error (log scale)', fontsize=14, fontweight='bold')
ax2.set_title('Question 7.2: Relative Error vs. Problem Size (n)\np=64, ntasks-per-node=4', 
              fontsize=15, fontweight='bold', pad=20)
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=1)
ax2.legend(fontsize=11, loc='best')

# Set x-axis ticks
ax2.set_xticks(n_values)
ax2.set_xticklabels(n_labels, fontsize=12)
ax2.tick_params(axis='y', labelsize=12)

# Add data labels with better positioning
for i, (x, y_original) in enumerate(zip(n_values, rel_error_p64)):
    if y_original > 0:
        label_text = f'{y_original:.2e}'
        ax2.annotate(label_text, xy=(x, rel_error_plot[i]), xytext=(0, 15),
                    textcoords='offset points', fontsize=10, fontweight='bold',
                    ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    else:
        label_text = '≈ 0'
        ax2.annotate(label_text, xy=(x, rel_error_plot[i]), xytext=(0, 15),
                    textcoords='offset points', fontsize=11, fontweight='bold',
                    ha='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

# Add interpretation box
interp_text = 'Accuracy improves with larger n:\nError decreases by ~12 orders of magnitude\nfrom n=10² to n=10⁸'
ax2.text(0.05, 0.05, interp_text,
        transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=1.5))

plt.tight_layout()
plt.savefig('question_7_2_error.png', dpi=300, bbox_inches='tight')
print("Question 7.2 plot saved: question_7_2_error.png")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print("\nQuestion 7.1 - Speedup Analysis:")
print("-" * 70)
print(f"• At n=10⁶: Speedup = 0.06x (SLOWDOWN - parallel overhead dominates)")
print(f"• At n=10⁸: Speedup = 5.29x (Good parallel efficiency)")
print(f"• Conclusion: Parallelization with p=64 only beneficial for large n")
print(f"  (need sufficient work to overcome communication/synchronization overhead)")

print("\nQuestion 7.2 - Relative Error Analysis:")
print("-" * 70)
print(f"• n=10²:  Relative error = 2.65e-06")
print(f"• n=10⁴:  Relative error = 2.65e-10  (10,000x improvement)")
print(f"• n=10⁶:  Relative error = 2.63e-14  (10,000x improvement)")
print(f"• n=10⁸:  Relative error ≈ 0         (near machine precision)")
print(f"• Conclusion: Monte Carlo method accuracy improves with √n samples")
