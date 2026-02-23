import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# Parse the results from the image
# Format: k, q, threads, time, qsort_time

results = {
    # k=12 results
    12: {
        0: (1, 0.0004, 0.0003),
        1: (2, 0.0003, 0.0003),
        2: (4, 0.0004, 0.0003),
        4: (16, 0.0016, 0.0003),
        6: (64, 0.0028, 0.0003),
        8: (256, 0.0118, 0.0003),
        10: (1024, 0.0511, 0.0003),
    },
    # k=20 results
    20: {
        0: (1, 0.1130, 0.1111),
        1: (2, 0.0621, 0.1120),
        2: (4, 0.0322, 0.1106),
        4: (16, 0.0135, 0.1088),
        6: (64, 0.0121, 0.1100),
        8: (256, 0.0399, 0.1108),
        10: (1024, 0.0863, 0.1117),
    },
    # k=28 results
    28: {
        0: (1, 39.5205, 41.3173),
        1: (2, 21.2306, 40.8916),
        2: (4, 10.9859, 41.3589),
        3: (8, 5.6725, 40.2936),
        4: (16, 2.9700, 41.3165),
        6: (64, 1.5411, 41.9233),
        8: (256, 1.7533, 40.7645),
    }
}

# Calculate speedup and efficiency for each k and q
def calculate_metrics(results_dict):
    metrics = {}
    for k in results_dict:
        metrics[k] = {}
        # Get baseline (q=0, single thread) time
        baseline_time = results_dict[k][0][1]
        
        for q in results_dict[k]:
            threads, time, qsort_time = results_dict[k][q]
            speedup = baseline_time / time
            efficiency = speedup / threads
            metrics[k][q] = {
                'threads': threads,
                'time': time,
                'speedup': speedup,
                'efficiency': efficiency,
                'qsort_time': qsort_time
            }
    return metrics

metrics = calculate_metrics(results)

# Print metrics table
print("="*80)
print("SPEEDUP AND EFFICIENCY ANALYSIS")
print("="*80)
for k in sorted(metrics.keys()):
    print(f"\nk = {k} (List Size = 2^{k} = {2**k:,})")
    print("-"*80)
    print(f"{'q':<4} {'Threads':<8} {'Time (s)':<12} {'Speedup':<10} {'Efficiency':<12}")
    print("-"*80)
    for q in sorted(metrics[k].keys()):
        m = metrics[k][q]
        print(f"{q:<4} {m['threads']:<8} {m['time']:<12.4f} {m['speedup']:<10.2f} {m['efficiency']:<12.4f}")

# Create comprehensive plots for Question 2
fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

k_values = [12, 20, 28]
colors = ['#2E86AB', '#A23B72', '#F18F01']
markers = ['o', 's', '^']

# Plot 1: Speedup vs Threads (log scale)
ax1 = fig.add_subplot(gs[0, 0])
for idx, k in enumerate(k_values):
    q_vals = sorted(metrics[k].keys())
    threads = [metrics[k][q]['threads'] for q in q_vals]
    speedups = [metrics[k][q]['speedup'] for q in q_vals]
    ax1.plot(threads, speedups, marker=markers[idx], color=colors[idx], 
             linewidth=2, markersize=8, label=f'k={k}')

# Ideal speedup line
max_threads = max([metrics[k][q]['threads'] for k in k_values for q in metrics[k].keys()])
ideal_threads = [2**i for i in range(11) if 2**i <= max_threads]
ax1.plot(ideal_threads, ideal_threads, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')

ax1.set_xscale('log', base=2)
ax1.set_yscale('log', base=2)
ax1.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
ax1.set_ylabel('Speedup', fontsize=11, fontweight='bold')
ax1.set_title('Speedup vs Number of Threads', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Plot 2: Efficiency vs Threads
ax2 = fig.add_subplot(gs[0, 1])
for idx, k in enumerate(k_values):
    q_vals = sorted(metrics[k].keys())
    threads = [metrics[k][q]['threads'] for q in q_vals]
    efficiencies = [metrics[k][q]['efficiency'] for q in q_vals]
    ax2.plot(threads, efficiencies, marker=markers[idx], color=colors[idx], 
             linewidth=2, markersize=8, label=f'k={k}')

ax2.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Efficiency')
ax2.set_xscale('log', base=2)
ax2.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
ax2.set_ylabel('Efficiency', fontsize=11, fontweight='bold')
ax2.set_title('Efficiency vs Number of Threads', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1.2)

# Plot 3: Execution Time vs Threads
ax3 = fig.add_subplot(gs[0, 2])
for idx, k in enumerate(k_values):
    q_vals = sorted(metrics[k].keys())
    threads = [metrics[k][q]['threads'] for q in q_vals]
    times = [metrics[k][q]['time'] for q in q_vals]
    ax3.plot(threads, times, marker=markers[idx], color=colors[idx], 
             linewidth=2, markersize=8, label=f'k={k}')

ax3.set_xscale('log', base=2)
ax3.set_yscale('log')
ax3.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
ax3.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
ax3.set_title('Execution Time vs Number of Threads', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)

# Plot 4: Speedup vs q (levels)
ax4 = fig.add_subplot(gs[1, 0])
for idx, k in enumerate(k_values):
    q_vals = sorted(metrics[k].keys())
    speedups = [metrics[k][q]['speedup'] for q in q_vals]
    ax4.plot(q_vals, speedups, marker=markers[idx], color=colors[idx], 
             linewidth=2, markersize=8, label=f'k={k}')

ax4.set_xlabel('q (log₂(threads))', fontsize=11, fontweight='bold')
ax4.set_ylabel('Speedup', fontsize=11, fontweight='bold')
ax4.set_title('Speedup vs q', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Plot 5: Efficiency vs q
ax5 = fig.add_subplot(gs[1, 1])
for idx, k in enumerate(k_values):
    q_vals = sorted(metrics[k].keys())
    efficiencies = [metrics[k][q]['efficiency'] for q in q_vals]
    ax5.plot(q_vals, efficiencies, marker=markers[idx], color=colors[idx], 
             linewidth=2, markersize=8, label=f'k={k}')

ax5.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect Efficiency')
ax5.set_xlabel('q (log₂(threads))', fontsize=11, fontweight='bold')
ax5.set_ylabel('Efficiency', fontsize=11, fontweight='bold')
ax5.set_title('Efficiency vs q', fontsize=12, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=10)

# Plot 6: Speedup comparison with qsort
ax6 = fig.add_subplot(gs[1, 2])
for idx, k in enumerate(k_values):
    q_vals = sorted(metrics[k].keys())
    threads = [metrics[k][q]['threads'] for q in q_vals]
    # Speedup relative to qsort
    qsort_speedups = [metrics[k][q]['qsort_time'] / metrics[k][q]['time'] for q in q_vals]
    ax6.plot(threads, qsort_speedups, marker=markers[idx], color=colors[idx], 
             linewidth=2, markersize=8, label=f'k={k}')

ax6.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, label='Equal to qsort')
ax6.set_xscale('log', base=2)
ax6.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
ax6.set_ylabel('Speedup vs qsort', fontsize=11, fontweight='bold')
ax6.set_title('Parallel Sort vs qsort Performance', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.legend(fontsize=10)

plt.suptitle('Problem 2: Parallel Merge Sort Performance Analysis', 
             fontsize=14, fontweight='bold', y=0.995)

plt.savefig('Q2_performance.png', dpi=300, bbox_inches='tight')
print("\n\nSaved: Q2_performance.png")

# Create focused plots for Question 3 - demonstrating best speedup cases
fig3 = plt.figure(figsize=(14, 5))

# Choose k=20 and k=28 as they show clear speedup
best_k_values = [20, 28]

# Plot 1: Detailed timing comparison for k=20
ax1 = fig3.add_subplot(1, 3, 1)
k = 20
q_vals = sorted(metrics[k].keys())
threads = [metrics[k][q]['threads'] for q in q_vals]
times = [metrics[k][q]['time'] for q in q_vals]
qsort_times = [metrics[k][q]['qsort_time'] for q in q_vals]

x_pos = np.arange(len(q_vals))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, times, width, label='Parallel Merge Sort', color='#2E86AB', alpha=0.8)
bars2 = ax1.bar(x_pos + width/2, qsort_times, width, label='qsort', color='#E63946', alpha=0.8)

ax1.set_xlabel('q (Number of threads = 2^q)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Time (seconds)', fontsize=11, fontweight='bold')
ax1.set_title(f'k={k} (List Size = {2**k:,})', fontsize=12, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([f'q={q}\n({2**q})' for q in q_vals], fontsize=9)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=8)

# Plot 2: Speedup and Efficiency for k=20
ax2 = fig3.add_subplot(1, 3, 2)
speedups = [metrics[20][q]['speedup'] for q in q_vals]
efficiencies = [metrics[20][q]['efficiency'] for q in q_vals]

ax2_twin = ax2.twinx()

line1 = ax2.plot(threads, speedups, marker='o', color='#2E86AB', 
                 linewidth=2.5, markersize=10, label='Speedup')
ax2.plot(threads, threads, 'k--', linewidth=1.5, alpha=0.5, label='Ideal Speedup')

line2 = ax2_twin.plot(threads, efficiencies, marker='s', color='#F18F01', 
                      linewidth=2.5, markersize=10, label='Efficiency')
ax2_twin.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, alpha=0.5)

ax2.set_xscale('log', base=2)
ax2.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
ax2.set_ylabel('Speedup', fontsize=11, fontweight='bold', color='#2E86AB')
ax2_twin.set_ylabel('Efficiency', fontsize=11, fontweight='bold', color='#F18F01')
ax2.set_title(f'k={k} Performance Metrics', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='y', labelcolor='#2E86AB')
ax2_twin.tick_params(axis='y', labelcolor='#F18F01')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper left', fontsize=10)

# Plot 3: Speedup and Efficiency for k=28
ax3 = fig3.add_subplot(1, 3, 3)
q_vals_28 = sorted(metrics[28].keys())
threads_28 = [metrics[28][q]['threads'] for q in q_vals_28]
speedups_28 = [metrics[28][q]['speedup'] for q in q_vals_28]
efficiencies_28 = [metrics[28][q]['efficiency'] for q in q_vals_28]

ax3_twin = ax3.twinx()

line1 = ax3.plot(threads_28, speedups_28, marker='o', color='#2E86AB', 
                 linewidth=2.5, markersize=10, label='Speedup')
ax3.plot(threads_28, threads_28, 'k--', linewidth=1.5, alpha=0.5, label='Ideal Speedup')

line2 = ax3_twin.plot(threads_28, efficiencies_28, marker='s', color='#F18F01', 
                      linewidth=2.5, markersize=10, label='Efficiency')
ax3_twin.axhline(y=1.0, color='k', linestyle=':', linewidth=1.5, alpha=0.5)

ax3.set_xscale('log', base=2)
ax3.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
ax3.set_ylabel('Speedup', fontsize=11, fontweight='bold', color='#2E86AB')
ax3_twin.set_ylabel('Efficiency', fontsize=11, fontweight='bold', color='#F18F01')
ax3.set_title(f'k={28} Performance Metrics', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.tick_params(axis='y', labelcolor='#2E86AB')
ax3_twin.tick_params(axis='y', labelcolor='#F18F01')

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc='upper left', fontsize=10)

plt.suptitle('Problem 3: Demonstrating Speedup with Optimal List Sizes', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

plt.savefig('Q2_speedup.png', dpi=300, bbox_inches='tight')
print("Saved: Q2_speedup.png")

# Create summary table for best results
print("\n" + "="*80)
print("PROBLEM 3: BEST SPEEDUP RESULTS")
print("="*80)
print("\nFor k=20 (List Size = 1,048,576):")
print("-"*80)
best_q_20 = max(metrics[20].keys(), key=lambda q: metrics[20][q]['speedup'])
print(f"Best speedup at q={best_q_20} ({2**best_q_20} threads):")
print(f"  Time: {metrics[20][best_q_20]['time']:.4f} seconds")
print(f"  Speedup: {metrics[20][best_q_20]['speedup']:.2f}x")
print(f"  Efficiency: {metrics[20][best_q_20]['efficiency']:.4f}")
print(f"  vs qsort: {metrics[20][best_q_20]['qsort_time'] / metrics[20][best_q_20]['time']:.2f}x faster")

print("\nFor k=28 (List Size = 268,435,456):")
print("-"*80)
best_q_28 = max(metrics[28].keys(), key=lambda q: metrics[28][q]['speedup'])
print(f"Best speedup at q={best_q_28} ({2**best_q_28} threads):")
print(f"  Time: {metrics[28][best_q_28]['time']:.4f} seconds")
print(f"  Speedup: {metrics[28][best_q_28]['speedup']:.2f}x")
print(f"  Efficiency: {metrics[28][best_q_28]['efficiency']:.4f}")
print(f"  vs qsort: {metrics[28][best_q_28]['qsort_time'] / metrics[28][best_q_28]['time']:.2f}x faster")

print("\n" + "="*80)
plt.show()
