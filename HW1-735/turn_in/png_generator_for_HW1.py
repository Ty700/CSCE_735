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
plt.show()
plt.tight_layout()
plt.savefig('error_vs_trials.png', dpi=300, bbox_inches='tight')

print("Plot saved successfully!")
print(f"\nError Analysis:")
print(f"n = 10^3:  error = {errors[0]:.2e}")
print(f"n = 10^9:  error = {errors[-1]:.2e}")
print(f"Error reduction factor: {errors[0]/errors[-1]:.1f}x")
print(f"Theoretical reduction (sqrt(10^6)): {np.sqrt(1e6):.1f}x")
