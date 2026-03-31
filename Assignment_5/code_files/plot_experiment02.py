"""
plot_experiment02.py  —  HPC Assignment 05, Experiment 02
=========================================================
Generates all required plots:

  Plot 1 (3 subplots): Speedup vs Threads
    — Mover WITH deletion  vs  WITHOUT deletion
    — Lab PC vs HPC  (4 curves per subplot)
    — One subplot per grid config

  Plot 2 (3 subplots): Execution Time vs Threads
    — interp / mover_plain / mover_with_del / total
    — One subplot per grid config

  Both for Approach 1 (Deferred) and Approach 2 (Immediate)

Usage:
    python plot_experiment02.py

Expected CSVs:
  Lab PC:
    results_exp02_grid1_250x100_deferred.csv
    results_exp02_grid1_250x100_immediate.csv
    ... (same pattern for grid2, grid3)

  HPC:
    results_exp02_grid1_250x100_deferred_hpc.csv
    ... etc.

CSV columns:
  Nx, Ny, np, num_threads,
  t_interp_s, t_mover_plain_s, t_mover_with_del_s, t_total_s,
  speedup_plain, speedup_with_del
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

GRIDS = [
    {"title": "Grid 1  (Nx=250,  Ny=100)",  "label": "grid1_250x100"},
    {"title": "Grid 2  (Nx=500,  Ny=200)",  "label": "grid2_500x200"},
    {"title": "Grid 3  (Nx=1000, Ny=400)",  "label": "grid3_1000x400"},
]

THREADS = [1, 2, 4, 8, 16]

def load(path):
    if not os.path.exists(path):
        print(f"  [skip] {path}")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

def ideal_speedup(threads):
    return threads

# ════════════════════════════════════════════════════════════════
# Helper: make one speedup subplot
# ════════════════════════════════════════════════════════════════
def speedup_subplot(ax, pc_def, pc_imm, hpc_def, hpc_imm, title, approach_label):
    # Ideal line
    ax.plot(THREADS, THREADS, 'k--', lw=1.2, label="Ideal speedup", alpha=0.5)

    def plot_curve(df, col, color, ls, marker, label):
        if df is None:
            return
        ax.plot(df["num_threads"], df[col],
                color=color, ls=ls, marker=marker,
                lw=2, ms=7, label=label)

    # Without deletion (plain)
    plot_curve(pc_def,  "speedup_plain",    "#1f77b4", "-",  "o", "Lab PC — No deletion")
    plot_curve(hpc_def, "speedup_plain",    "#d62728", "-",  "o", "HPC — No deletion")
    # With deletion
    plot_curve(pc_def,  "speedup_with_del", "#1f77b4", "--", "s", "Lab PC — With deletion")
    plot_curve(hpc_def, "speedup_with_del", "#d62728", "--", "s", "HPC — With deletion")

    ax.set_title(f"{title}\n({approach_label})", fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Threads", fontsize=10)
    ax.set_ylabel("Speedup", fontsize=10)
    ax.set_xticks(THREADS)
    ax.set_xticklabels([str(t) for t in THREADS])
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8)

# ════════════════════════════════════════════════════════════════
# PLOT 1a — Speedup: Deferred Approach
# ════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle(
    "Experiment 02 — Speedup vs Threads\n"
    "Approach 1: Deferred Insertion  |  np = 14,000,000  |  Maxiter = 10",
    fontsize=13, fontweight="bold", y=1.02
)
for ax, g in zip(axes1, GRIDS):
    pc_def  = load(f"results_exp02_{g['label']}_deferred.csv")
    hpc_def = load(f"results_exp02_{g['label']}_deferred_hpc.csv")
    speedup_subplot(ax, pc_def, None, hpc_def, None, g["title"], "Deferred Insertion")

plt.tight_layout()
plt.savefig("plot_exp02_speedup_deferred.png", dpi=150, bbox_inches="tight")
plt.savefig("plot_exp02_speedup_deferred.pdf", bbox_inches="tight")
print("Saved: plot_exp02_speedup_deferred.png/.pdf")

# ════════════════════════════════════════════════════════════════
# PLOT 1b — Speedup: Immediate Approach
# ════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle(
    "Experiment 02 — Speedup vs Threads\n"
    "Approach 2: Immediate Replacement  |  np = 14,000,000  |  Maxiter = 10",
    fontsize=13, fontweight="bold", y=1.02
)
for ax, g in zip(axes2, GRIDS):
    pc_imm  = load(f"results_exp02_{g['label']}_immediate.csv")
    hpc_imm = load(f"results_exp02_{g['label']}_immediate_hpc.csv")
    speedup_subplot(ax, None, pc_imm, None, hpc_imm, g["title"], "Immediate Replacement")

plt.tight_layout()
plt.savefig("plot_exp02_speedup_immediate.png", dpi=150, bbox_inches="tight")
plt.savefig("plot_exp02_speedup_immediate.pdf", bbox_inches="tight")
print("Saved: plot_exp02_speedup_immediate.png/.pdf")

# ════════════════════════════════════════════════════════════════
# PLOT 2 — Execution Time vs Threads (both approaches, Lab PC)
# ════════════════════════════════════════════════════════════════
for approach, approach_label, suffix in [
    ("deferred",  "Deferred Insertion",   "deferred"),
    ("immediate", "Immediate Replacement","immediate"),
]:
    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
    fig3.suptitle(
        f"Experiment 02 — Execution Time vs Threads  ({approach_label})\n"
        f"Lab PC  |  np = 14,000,000  |  Maxiter = 10",
        fontsize=13, fontweight="bold", y=1.02
    )
    for ax, g in zip(axes3, GRIDS):
        df = load(f"results_exp02_{g['label']}_{suffix}.csv")
        if df is None:
            continue

        ax.plot(df["num_threads"], df["t_interp_s"],
                color="#2ca02c", ls="-",  marker="^", lw=2, ms=6, label="Interpolation")
        ax.plot(df["num_threads"], df["t_mover_plain_s"],
                color="#1f77b4", ls="-",  marker="o", lw=2, ms=6, label="Mover (no deletion)")
        ax.plot(df["num_threads"], df["t_mover_with_del_s"],
                color="#d62728", ls="--", marker="s", lw=2, ms=6, label="Mover (with deletion)")
        ax.plot(df["num_threads"], df["t_total_s"],
                color="#7f7f7f", ls="-.", marker="D", lw=2, ms=6, label="Total")

        ax.set_title(g["title"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Number of Threads", fontsize=10)
        ax.set_ylabel("Execution Time  [s]", fontsize=10)
        ax.set_xticks(THREADS)
        ax.set_xticklabels([str(t) for t in THREADS])
        ax.grid(True, ls="--", lw=0.5, alpha=0.6)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fname = f"plot_exp02_time_{suffix}"
    plt.savefig(fname + ".png", dpi=150, bbox_inches="tight")
    plt.savefig(fname + ".pdf", bbox_inches="tight")
    print(f"Saved: {fname}.png/.pdf")

# ════════════════════════════════════════════════════════════════
# PLOT 3 — Combined: Lab PC vs HPC speedup on same axes
#           Both approaches overlaid (all 4 curves per grid)
# ════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
fig4.suptitle(
    "Experiment 02 — Combined Speedup: Lab PC vs HPC\n"
    "Deferred & Immediate  |  With vs Without Deletion  |  np = 14,000,000",
    fontsize=12, fontweight="bold", y=1.02
)
for ax, g in zip(axes4, GRIDS):
    ax.plot(THREADS, THREADS, 'k--', lw=1, alpha=0.4, label="Ideal")

    def try_plot(path, col, color, ls, marker, label):
        df = load(path)
        if df is None:
            return
        ax.plot(df["num_threads"], df[col],
                color=color, ls=ls, marker=marker,
                lw=2, ms=6, label=label)

    try_plot(f"results_exp02_{g['label']}_deferred.csv",
             "speedup_with_del", "#1f77b4", "-",  "o", "PC Deferred")
    try_plot(f"results_exp02_{g['label']}_immediate.csv",
             "speedup_with_del", "#1f77b4", "--", "s", "PC Immediate")
    try_plot(f"results_exp02_{g['label']}_deferred_hpc.csv",
             "speedup_with_del", "#d62728", "-",  "o", "HPC Deferred")
    try_plot(f"results_exp02_{g['label']}_immediate_hpc.csv",
             "speedup_with_del", "#d62728", "--", "s", "HPC Immediate")

    ax.set_title(g["title"], fontsize=11, fontweight="bold")
    ax.set_xlabel("Number of Threads", fontsize=10)
    ax.set_ylabel("Speedup", fontsize=10)
    ax.set_xticks(THREADS)
    ax.set_xticklabels([str(t) for t in THREADS])
    ax.grid(True, ls="--", lw=0.5, alpha=0.6)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("plot_exp02_combined_speedup.png", dpi=150, bbox_inches="tight")
plt.savefig("plot_exp02_combined_speedup.pdf", bbox_inches="tight")
print("Saved: plot_exp02_combined_speedup.png/.pdf")

print("\nAll Experiment 02 plots saved.")
plt.show()
