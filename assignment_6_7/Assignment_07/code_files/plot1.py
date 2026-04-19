import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
local = pd.read_csv("results.csv")
cluster = pd.read_csv("results_cluster.csv")

local.columns = local.columns.str.strip()
cluster.columns = cluster.columns.str.strip()

files = local['File'].unique()

# =========================
# HELPERS
# =========================
def compute_metrics(data):
    data = data.sort_values(by="Threads").copy()
    base = data[data['Threads'] == 2]

    if base.empty:
        return None

    base_time = base['Total'].values[0]

    data['Speedup'] = base_time / data['Total']
    data['Efficiency'] = data['Speedup'] / data['Threads']
    data['Normalized'] = data['Total'] / base_time

    return data

# =========================
# STYLE
# =========================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
})

# =========================
# LOOP
# =========================
for f in files:

    d_local = local[local['File'] == f].sort_values(by="Threads")
    d_cluster = cluster[cluster['File'] == f].sort_values(by="Threads")

    d_local_m = compute_metrics(d_local)
    d_cluster_m = compute_metrics(d_cluster)

    # =========================
    # 1. TIME
    # =========================
    plt.figure(figsize=(7,5))

    plt.plot(d_local['Threads'].to_numpy(),
             d_local['Total'].to_numpy(),
             marker='o', label="Local")

    plt.plot(d_cluster['Threads'].to_numpy(),
             d_cluster['Total'].to_numpy(),
             linestyle='--', marker='s', label="Cluster")

    plt.xlabel("Threads")
    plt.ylabel("Time (s)")
    plt.title(f"{f}: Execution Time")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{f}_time.png")
    plt.close()

    # =========================
    # 2. SPEEDUP
    # =========================
    if d_local_m is not None and d_cluster_m is not None:
        plt.figure(figsize=(7,5))

        plt.plot(d_local_m['Threads'].to_numpy(),
                 d_local_m['Speedup'].to_numpy(),
                 marker='o', label="Local")

        plt.plot(d_cluster_m['Threads'].to_numpy(),
                 d_cluster_m['Speedup'].to_numpy(),
                 linestyle='--', marker='s', label="Cluster")

        threads = d_local_m['Threads'].to_numpy()
        plt.plot(threads, threads, linestyle=':', label="Ideal")

        plt.xlabel("Threads")
        plt.ylabel("Speedup")
        plt.title(f"{f}: Speedup")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{f}_speedup.png")
        plt.close()

    # =========================
    # 3. EFFICIENCY
    # =========================
    if d_local_m is not None and d_cluster_m is not None:
        plt.figure(figsize=(7,5))

        plt.plot(d_local_m['Threads'].to_numpy(),
                 d_local_m['Efficiency'].to_numpy(),
                 marker='o', label="Local")

        plt.plot(d_cluster_m['Threads'].to_numpy(),
                 d_cluster_m['Efficiency'].to_numpy(),
                 linestyle='--', marker='s', label="Cluster")

        plt.xlabel("Threads")
        plt.ylabel("Efficiency")
        plt.title(f"{f}: Efficiency")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{f}_efficiency.png")
        plt.close()

    # =========================
    # 4. PHASE CONTRIBUTION
    # =========================
    plt.figure(figsize=(7,5))

    plt.plot(d_local['Threads'].to_numpy(),
             d_local['Interpolation'].to_numpy(),
             marker='o', label='Interpolation')

    plt.plot(d_local['Threads'].to_numpy(),
             d_local['Mover'].to_numpy(),
             marker='o', label='Mover')

    plt.plot(d_local['Threads'].to_numpy(),
             (d_local['Normalization'] + d_local['Denormalization']).to_numpy(),
             marker='o', label='Norm+Denorm')

    plt.xlabel("Threads")
    plt.ylabel("Time (s)")
    plt.title(f"{f}: Phase Contribution (Local)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{f}_phase.png")
    plt.close()

    # =========================
    # 5. NORMALIZED TIME
    # =========================
    if d_local_m is not None:
        plt.figure(figsize=(7,5))

        plt.plot(d_local_m['Threads'].to_numpy(),
                 d_local_m['Normalized'].to_numpy(),
                 marker='o', label="Local")

        plt.plot(d_cluster_m['Threads'].to_numpy(),
                 d_cluster_m['Normalized'].to_numpy(),
                 linestyle='--', marker='s', label="Cluster")

        plt.xlabel("Threads")
        plt.ylabel("Normalized Time")
        plt.title(f"{f}: Normalized Time")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{f}_normalized.png")
        plt.close()

    # =========================
    # 6. INTERPOLATION vs MOVER RATIO
    # =========================
    plt.figure(figsize=(7,5))

    ratio_local = d_local['Interpolation'] / d_local['Mover']
    plt.plot(d_local['Threads'].to_numpy(),
             ratio_local.to_numpy(),
             marker='o', label="Local Ratio")

    plt.xlabel("Threads")
    plt.ylabel("Interpolation / Mover")
    plt.title(f"{f}: Bottleneck Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{f}_ratio.png")
    plt.close()

print("✅ All advanced analytical plots generated!")