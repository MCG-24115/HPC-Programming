import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
local = pd.read_csv("results.csv")
cluster = pd.read_csv("results_cluster.csv")

# Clean column names
local.columns = local.columns.str.strip()
cluster.columns = cluster.columns.str.strip()

files = local['File'].unique()

# =========================
# HELPER
# =========================
def compute_metrics(data):
    data = data.sort_values(by="Threads").copy()
    base = data[data['Threads'] == 2]

    if base.empty:
        return None

    base_time = base['Total'].values[0]
    data['Speedup'] = base_time / data['Total']
    data['Efficiency'] = data['Speedup'] / data['Threads']

    return data

# =========================
# STYLE SETTINGS
# =========================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11
})

# =========================
# LOOP OVER FILES
# =========================
for f in files:

    d_local = local[local['File'] == f].sort_values(by="Threads")
    d_cluster = cluster[cluster['File'] == f].sort_values(by="Threads")

    d_local_m = compute_metrics(d_local)
    d_cluster_m = compute_metrics(d_cluster)

    # =========================
    # 1. TIME PLOT
    # =========================
    plt.figure(figsize=(7,5))

    if not d_local.empty:
        plt.plot(d_local['Threads'].to_numpy(),
                 d_local['Total'].to_numpy(),
                 marker='o',
                 label="Local")

    if not d_cluster.empty:
        plt.plot(d_cluster['Threads'].to_numpy(),
                 d_cluster['Total'].to_numpy(),
                 linestyle='--',
                 marker='s',
                 label="Cluster")

    # Annotate best (minimum time)
    if not d_local.empty:
        idx = d_local['Total'].idxmin()
        plt.annotate("Best Local",
                     (d_local.loc[idx, 'Threads'], d_local.loc[idx, 'Total']),
                     textcoords="offset points", xytext=(0,8), ha='center')

    if not d_cluster.empty:
        idx = d_cluster['Total'].idxmin()
        plt.annotate("Best Cluster",
                     (d_cluster.loc[idx, 'Threads'], d_cluster.loc[idx, 'Total']),
                     textcoords="offset points", xytext=(0,-12), ha='center')

    plt.xlabel("Threads")
    plt.ylabel("Execution Time (s)")
    plt.title(f"{f}: Execution Time vs Threads")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{f}_time.png")
    plt.close()

    # =========================
    # 2. SPEEDUP PLOT
    # =========================
    plt.figure(figsize=(7,5))

    if d_local_m is not None:
        plt.plot(d_local_m['Threads'].to_numpy(),
                 d_local_m['Speedup'].to_numpy(),
                 marker='o',
                 label="Local")

    if d_cluster_m is not None:
        plt.plot(d_cluster_m['Threads'].to_numpy(),
                 d_cluster_m['Speedup'].to_numpy(),
                 linestyle='--',
                 marker='s',
                 label="Cluster")

    # Ideal line
    if not d_local.empty:
        threads = d_local['Threads'].to_numpy()
        plt.plot(threads, threads, linestyle=':', label="Ideal")

    plt.xlabel("Threads")
    plt.ylabel("Speedup")
    plt.title(f"{f}: Speedup vs Threads")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{f}_speedup.png")
    plt.close()

    # =========================
    # 3. EFFICIENCY PLOT
    # =========================
    plt.figure(figsize=(7,5))

    if d_local_m is not None:
        plt.plot(d_local_m['Threads'].to_numpy(),
                 d_local_m['Efficiency'].to_numpy(),
                 marker='o',
                 label="Local")

    if d_cluster_m is not None:
        plt.plot(d_cluster_m['Threads'].to_numpy(),
                 d_cluster_m['Efficiency'].to_numpy(),
                 linestyle='--',
                 marker='s',
                 label="Cluster")

    # Annotate drop point
    if d_local_m is not None:
        last = d_local_m.iloc[-1]
        plt.annotate("Efficiency Drop",
                     (last['Threads'], last['Efficiency']),
                     textcoords="offset points", xytext=(0,-10), ha='center')

    plt.xlabel("Threads")
    plt.ylabel("Efficiency")
    plt.title(f"{f}: Efficiency vs Threads")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{f}_efficiency.png")
    plt.close()

print("✅ Clean, annotated, report-ready plots generated!")