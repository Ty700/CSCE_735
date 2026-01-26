import matplotlib.pyplot as plt
import numpy as np

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
 
