"""
plot_experiment01.py  —  HPC Assignment 05, Experiment 01
=========================================================
Generates ALL plots required by the assignment:

  Plot 1-3  : Execution time vs particle count (log-log)
              — 4 lines per grid: Lab PC / HPC × Deferred / Immediate
              — one plot per grid config  (3 total, as required)

  Plot 4    : Per-particle execution time vs PPC (particles per cell)

  Plot 5    : Memory requirement vs particle count

  Plot 6    : Particle distribution verification
              (scatter + histograms for x and y)

Usage:
  python plot_experiment01.py

Expected CSVs (Lab PC):
  results_grid1_250x100_deferred.csv   results_grid1_250x100_immediate.csv
  results_grid2_500x200_deferred.csv   results_grid2_500x200_immediate.csv
  results_grid3_1000x400_deferred.csv  results_grid3_1000x400_immediate.csv
  distribution_grid1_250x100.csv  (or any one grid — for verification)

HPC versions: same names with _hpc suffix before .csv
  e.g.  results_grid1_250x100_deferred_hpc.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Helpers ──────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        print(f"  [skip] {path} not found")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df

PARTICLE_COUNTS = [1e2, 1e4, 1e6, 1e8, 1e9]

GRIDS = [
    {"title": "Grid 1  (Nx=250,  Ny=100)",
     "label": "grid1_250x100"},
    {"title": "Grid 2  (Nx=500,  Ny=200)",
     "label": "grid2_500x200"},
    {"title": "Grid 3  (Nx=1000, Ny=400)",
     "label": "grid3_1000x400"},
]

STYLES = {
    "pc_def":  dict(color="#1f77b4", ls="-",  marker="o", lw=2, ms=6),
    "pc_imm":  dict(color="#1f77b4", ls="--", marker="s", lw=2, ms=6),
    "hpc_def": dict(color="#d62728", ls="-",  marker="o", lw=2, ms=6),
    "hpc_imm": dict(color="#d62728", ls="--", marker="s", lw=2, ms=6),
}

def xtick_fmt(ax):
    ax.set_xticks(PARTICLE_COUNTS)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda v, _: r"$10^{%d}$" % int(round(np.log10(v)))
        )
    )

def add_loglog_line(ax, df, ycol, style, label):
    if df is None:
        return
    mask = df[ycol] > 0
    ax.loglog(df.loc[mask, "np"], df.loc[mask, ycol],
              label=label, **style)

# ════════════════════════════════════════════════════════════════
# PLOTS 1–3 : Execution time vs particle count  (3 grids)
# ════════════════════════════════════════════════════════════════
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))
fig1.suptitle(
    "Experiment 01 — Total Execution Time vs Particle Count\n"
    "Deferred Insertion vs Immediate Replacement  |  Lab PC vs HPC  |  Maxiter=10",
    fontsize=13, fontweight="bold", y=1.02
)

for ax, g in zip(axes1, GRIDS):
    pc_def  = load(f"results_{g['label']}_deferred.csv")
    pc_imm  = load(f"results_{g['label']}_immediate.csv")
    hpc_def = load(f"results_{g['label']}_deferred_hpc.csv")
    hpc_imm = load(f"results_{g['label']}_immediate_hpc.csv")

    add_loglog_line(ax, pc_def,  "total_s", STYLES["pc_def"],  "Lab PC — Deferred")
    add_loglog_line(ax, pc_imm,  "total_s", STYLES["pc_imm"],  "Lab PC — Immediate")
    add_loglog_line(ax, hpc_def, "total_s", STYLES["hpc_def"], "HPC — Deferred")
    add_loglog_line(ax, hpc_imm, "total_s", STYLES["hpc_imm"], "HPC — Immediate")

    ax.set_title(g["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Particles  (log scale)", fontsize=10)
    ax.set_ylabel("Total Execution Time  [s]  (log scale)", fontsize=10)
    xtick_fmt(ax)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    ax.minorticks_on()
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot1_execution_time.png", dpi=150, bbox_inches="tight")
plt.savefig("plot1_execution_time.pdf", bbox_inches="tight")
print("Saved: plot1_execution_time.png/.pdf")

# ════════════════════════════════════════════════════════════════
# PLOT 4 : Per-particle execution time vs PPC
# ════════════════════════════════════════════════════════════════
fig4, axes4 = plt.subplots(1, 3, figsize=(18, 6))
fig4.suptitle(
    "Per-Particle Execution Time vs Particles Per Cell (PPC)\n"
    "Deferred vs Immediate  |  Lab PC",
    fontsize=13, fontweight="bold", y=1.02
)

for ax, g in zip(axes4, GRIDS):
    pc_def = load(f"results_{g['label']}_deferred.csv")
    pc_imm = load(f"results_{g['label']}_immediate.csv")

    if pc_def is not None:
        ax.loglog(pc_def["ppc"], pc_def["time_per_particle_s"],
                  label="Deferred", **STYLES["pc_def"])
    if pc_imm is not None:
        ax.loglog(pc_imm["ppc"], pc_imm["time_per_particle_s"],
                  label="Immediate", **STYLES["pc_imm"])

    ax.set_title(g["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Particles Per Cell  (PPC, log scale)", fontsize=10)
    ax.set_ylabel("Time per Particle  [s]  (log scale)", fontsize=10)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    ax.minorticks_on()
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot4_per_particle_time.png", dpi=150, bbox_inches="tight")
plt.savefig("plot4_per_particle_time.pdf", bbox_inches="tight")
print("Saved: plot4_per_particle_time.png/.pdf")

# ════════════════════════════════════════════════════════════════
# PLOT 5 : Memory requirement vs particle count
# ════════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(1, 3, figsize=(18, 6))
fig5.suptitle(
    "Memory Requirement vs Particle Count\n"
    "Particles Array vs Mesh Grid vs Total",
    fontsize=13, fontweight="bold", y=1.02
)

for ax, g in zip(axes5, GRIDS):
    df = load(f"results_{g['label']}_deferred.csv")
    if df is not None:
        ax.loglog(df["np"], df["mem_particles_MB"],
                  label="Particles array",
                  color="#1f77b4", ls="-", marker="o", lw=2, ms=6)
        ax.loglog(df["np"], df["mem_mesh_MB"],
                  label="Mesh grid",
                  color="#2ca02c", ls="--", marker="s", lw=2, ms=6)
        ax.loglog(df["np"], df["mem_total_MB"],
                  label="Total",
                  color="#d62728", ls="-.", marker="^", lw=2, ms=6)

    ax.set_title(g["title"], fontsize=12, fontweight="bold")
    ax.set_xlabel("Number of Particles  (log scale)", fontsize=10)
    ax.set_ylabel("Memory  [MB]  (log scale)", fontsize=10)
    xtick_fmt(ax)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.6)
    ax.minorticks_on()
    ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig("plot5_memory.png", dpi=150, bbox_inches="tight")
plt.savefig("plot5_memory.pdf", bbox_inches="tight")
print("Saved: plot5_memory.png/.pdf")

# ════════════════════════════════════════════════════════════════
# PLOT 6 : Particle distribution verification
# (scatter plot + x-histogram + y-histogram)
# Uses distribution_<grid>.csv from largest np run
# ════════════════════════════════════════════════════════════════
dist_files = [
    ("distribution_grid1_250x100.csv",  "Grid 1 (250×100)"),
    ("distribution_grid2_500x200.csv",  "Grid 2 (500×200)"),
    ("distribution_grid3_1000x400.csv", "Grid 3 (1000×400)"),
]

for dist_path, title in dist_files:
    dist_df = load(dist_path)
    if dist_df is None:
        continue

    fig6, axes6 = plt.subplots(1, 3, figsize=(15, 5))
    fig6.suptitle(
        f"Particle Distribution Verification — {title}",
        fontsize=13, fontweight="bold"
    )

    # Scatter
    ax = axes6[0]
    ax.scatter(dist_df["x"], dist_df["y"],
               s=0.5, alpha=0.3, color="#1f77b4", rasterized=True)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("x", fontsize=11); ax.set_ylabel("y", fontsize=11)
    ax.set_title("Particle Positions", fontsize=11)
    ax.set_aspect("equal")

    # X histogram
    ax = axes6[1]
    ax.hist(dist_df["x"], bins=50, color="#1f77b4",
            edgecolor="white", linewidth=0.3)
    ax.axhline(len(dist_df) / 50, color="red", ls="--",
               lw=1.5, label="Expected uniform")
    ax.set_xlabel("x coordinate", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("x-coordinate distribution", fontsize=11)
    ax.legend(fontsize=9)

    # Y histogram
    ax = axes6[2]
    ax.hist(dist_df["y"], bins=50, color="#2ca02c",
            edgecolor="white", linewidth=0.3)
    ax.axhline(len(dist_df) / 50, color="red", ls="--",
               lw=1.5, label="Expected uniform")
    ax.set_xlabel("y coordinate", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("y-coordinate distribution", fontsize=11)
    ax.legend(fontsize=9)

    plt.tight_layout()
    safe = title.replace(" ", "_").replace("(", "").replace(")", "").replace("×", "x")
    fname = f"plot6_distribution_{safe}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    print(f"Saved: {fname}")

# ════════════════════════════════════════════════════════════════
# TABLE : Memory & FLOPs summary  (printed to console + saved CSV)
# ════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("MEMORY & FLOPs SUMMARY TABLE")
print("="*80)
print(f"{'Grid':<20} {'np':>12} {'Mem_part(MB)':>14} {'Mem_mesh(MB)':>14} "
      f"{'Mem_total(MB)':>14} {'FLOPs_total':>14} {'PPC':>8}")
print("-"*80)

for g in GRIDS:
    df = load(f"results_{g['label']}_deferred.csv")
    if df is None:
        continue
    for _, row in df.iterrows():
        print(f"{g['label']:<20} {int(row['np']):>12} "
              f"{row['mem_particles_MB']:>14.3f} "
              f"{row['mem_mesh_MB']:>14.3f} "
              f"{row['mem_total_MB']:>14.3f} "
              f"{row['flops_total']:>14.3e} "
              f"{row['ppc']:>8.4f}")

print("\nAll plots saved.")
plt.show()
