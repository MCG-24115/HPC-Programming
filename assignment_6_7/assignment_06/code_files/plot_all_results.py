import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================================
# LOAD DATA
# ================================

serial = pd.read_csv("serial.csv")
parallel = pd.read_csv("parallel.csv")

# Merge SERIAL cluster (in case split)
sc1 = pd.read_csv("serial_cluster.csv")
try:
    sc2 = pd.read_csv("serial_cluster2.csv")
    serial_cluster = pd.concat([sc1, sc2], ignore_index=True)
except:
    serial_cluster = sc1

# Merge PARALLEL cluster (split files)
pc1 = pd.read_csv("parallel_cluster.csv")
pc2 = pd.read_csv("parallel_cluster2.csv")
parallel_cluster = pd.concat([pc1, pc2], ignore_index=True)

configs = ["A", "B", "C", "D", "E"]

# ================================
# STYLE
# ================================
plt.style.use("default")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f5f5f5",
    "axes.edgecolor": "black",
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "legend.fontsize": 12
})

# ================================
# LOOP CONFIGS
# ================================
for cfg in configs:

    # -------------------------------
    # SAFE EXTRACTION
    # -------------------------------
    try:
        s_pc = float(serial[serial["Config"] == cfg]["Time"].iloc[0])
        s_cl = float(serial_cluster[serial_cluster["Config"] == cfg]["Time"].iloc[0])
    except IndexError:
        print(f"⚠️ Missing serial data for config {cfg}, skipping...")
        continue

    p_pc = parallel[parallel["Config"] == cfg].sort_values("Threads")
    p_cl = parallel_cluster[parallel_cluster["Config"] == cfg].sort_values("Threads")

    if p_pc.empty or p_cl.empty:
        print(f"⚠️ Missing parallel data for config {cfg}, skipping...")
        continue

    threads = p_pc["Threads"].values

    time_pc = p_pc["Time"].values
    time_cl = p_cl["Time"].values

    # -------------------------------
    # METRICS
    # -------------------------------
    speedup_pc = s_pc / time_pc
    speedup_cl = s_cl / time_cl

    eff_pc = speedup_pc / threads
    eff_cl = speedup_cl / threads

    # ================================
    # 1️⃣ TIME vs CORES
    # ================================
    plt.figure(figsize=(9, 6))

    plt.plot(threads, time_pc, 'o-', linewidth=2, label="PC Parallel")
    plt.plot(threads, time_cl, 's-', linewidth=2, label="Cluster Parallel")

    plt.axhline(y=s_pc, linestyle='--', label="PC Serial")
    plt.axhline(y=s_cl, linestyle='-.', label="Cluster Serial")

    for i in range(len(threads)):
        plt.text(threads[i], time_pc[i], f"{time_pc[i]:.3f}")
        plt.text(threads[i], time_cl[i], f"{time_cl[i]:.3f}")

    plt.title(f"Execution Time vs Cores (Config {cfg})")
    plt.xlabel("Cores")
    plt.ylabel("Time (seconds)")
    plt.xticks(threads)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"time_comparison_{cfg}.png", dpi=300)
    plt.close()

    # ================================
    # 2️⃣ SPEEDUP vs CORES
    # ================================
    plt.figure(figsize=(9, 6))

    plt.plot(threads, speedup_pc, 'o-', linewidth=2, label="PC Speedup")
    plt.plot(threads, speedup_cl, 's-', linewidth=2, label="Cluster Speedup")

    plt.plot(threads, threads, '--', label="Ideal")

    for i in range(len(threads)):
        plt.text(threads[i], speedup_pc[i], f"{speedup_pc[i]:.2f}")
        plt.text(threads[i], speedup_cl[i], f"{speedup_cl[i]:.2f}")

    plt.title(f"Speedup vs Cores (Config {cfg})")
    plt.xlabel("Cores")
    plt.ylabel("Speedup")

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"speedup_comparison_{cfg}.png", dpi=300)
    plt.close()

    # ================================
    # 3️⃣ EFFICIENCY vs CORES
    # ================================
    plt.figure(figsize=(9, 6))

    plt.plot(threads, eff_pc, 'o-', linewidth=2, label="PC Efficiency")
    plt.plot(threads, eff_cl, 's-', linewidth=2, label="Cluster Efficiency")

    for i in range(len(threads)):
        plt.text(threads[i], eff_pc[i], f"{eff_pc[i]:.2f}")
        plt.text(threads[i], eff_cl[i], f"{eff_cl[i]:.2f}")

    plt.title(f"Efficiency vs Cores (Config {cfg})")
    plt.xlabel("Cores")
    plt.ylabel("Efficiency")
    plt.ylim(0, 1.1)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"efficiency_comparison_{cfg}.png", dpi=300)
    plt.close()

print("\n✅ ALL PLOTS GENERATED (PC vs CLUSTER)!")