"""
HPC Assignment 08 - Parallel Interpolation with Particle Mover (MPI+OpenMP)
Performance Analysis & Visualization Script
============================================================
Generates all required plots:
  1. Execution Time vs. Total Cores (per config)
  2. Speedup vs. Total Cores (per config)
  3. Parallel Efficiency vs. Total Cores (per config)
  4. Phase Breakdown: Interpolation vs. Mover (per config)
  5. Combined Speedup comparison across all 5 configs
  6. Heatmap: Execution time for different MPI×OMP combos (per config)

Usage:
    python plot_hpc_results.py

Place timings_a.csv through timings_e.csv in the same directory,
or update CSV_FILES below.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CSV_FILES = [
    "timings_a.csv",
    "timings_b.csv",
    "timings_c.csv",
    "timings_d.csv",
    "timings_e.csv",
]

CONFIG_LABELS = [
    "Config A\nNx=250, Ny=100\n0.9M pts",
    "Config B\nNx=250, Ny=100\n5M pts",
    "Config C\nNx=500, Ny=200\n3.6M pts",
    "Config D\nNx=500, Ny=200\n20M pts",
    "Config E\nNx=1000, Ny=400\n14M pts",
]

CONFIG_SHORT = ["A (0.9M)", "B (5M)", "C (3.6M)", "D (20M)", "E (14M)"]

# Publication-quality color palette
PALETTE = ["#E63946", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
PHASE_COLORS = {
    "interp_time": "#E63946",
    "mover_time":  "#2196F3",
    "norm_time":   "#FFC107",
    "denorm_time": "#4CAF50",
}

STYLE = {
    "figure.facecolor": "#0D1117",
    "axes.facecolor":   "#161B22",
    "axes.edgecolor":   "#30363D",
    "axes.labelcolor":  "#C9D1D9",
    "axes.titlecolor":  "#F0F6FC",
    "text.color":       "#C9D1D9",
    "xtick.color":      "#8B949E",
    "ytick.color":      "#8B949E",
    "grid.color":       "#21262D",
    "grid.linewidth":   0.8,
    "legend.facecolor": "#21262D",
    "legend.edgecolor": "#30363D",
    "font.family":      "DejaVu Sans",
    "font.size":        11,
}

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def load_data(path):
    """Load a timing CSV, adding a 'total_cores' column."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["total_cores"] = df["mpi_ranks"] * df["omp_threads"]
    return df


def best_per_cores(df):
    """For each total_cores value keep the row with minimum total_time."""
    return df.loc[df.groupby("total_cores")["total_time"].idxmin()].sort_values("total_cores")


def serial_time(df):
    """Estimate serial time: best total_time at total_cores == 1 (or min cores)."""
    min_cores = df["total_cores"].min()
    return df[df["total_cores"] == min_cores]["total_time"].min()


def apply_dark_style():
    plt.rcParams.update(STYLE)


def finish_ax(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.minorticks_on()
    if legend:
        ax.legend(fontsize=9, framealpha=0.8)


def add_watermark(fig):
    fig.text(0.99, 0.01, "HPC Assignment 08 | DA-IICT",
             ha="right", va="bottom", fontsize=8,
             color="#444C56", style="italic")

# ──────────────────────────────────────────────────────────────
# PLOT 1 – EXECUTION TIME vs TOTAL CORES  (one per config)
# ──────────────────────────────────────────────────────────────

def plot_exec_time_per_config(datasets):
    apply_dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("Execution Time vs. Total Cores\n(best MPI×OMP combination per core count)",
                 fontsize=15, fontweight="bold", y=1.01)

    for idx, (df, label, color) in enumerate(zip(datasets, CONFIG_LABELS, PALETTE)):
        ax = axes[idx]
        if df is None:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes, color="#8B949E")
            ax.set_title(label.replace("\n", " "), fontsize=11)
            continue

        best = best_per_cores(df)
        cores = best["total_cores"].values
        times = best["total_time"].values

        ax.plot(cores, times, "o-", color=color, linewidth=2.2, markersize=7,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2,
                label="Best combo")

        # annotate each point
        for c, t in zip(cores, times):
            ax.annotate(f"{t:.3f}s", xy=(c, t),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7.5, color="#C9D1D9")

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(cores)
        finish_ax(ax, label.replace("\n", " | "), "Total Cores (log₂)", "Time (s)", legend=False)

    axes[-1].set_visible(False)
    fig.tight_layout()
    add_watermark(fig)
    fname = "plot1_exec_time_per_config.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname

# ──────────────────────────────────────────────────────────────
# PLOT 2 – SPEEDUP vs TOTAL CORES  (one per config)
# ──────────────────────────────────────────────────────────────

def plot_speedup_per_config(datasets):
    apply_dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("Speedup vs. Total Cores\n(S = T_serial / T_parallel  |  Ideal = linear)",
                 fontsize=15, fontweight="bold", y=1.01)

    for idx, (df, label, color) in enumerate(zip(datasets, CONFIG_LABELS, PALETTE)):
        ax = axes[idx]
        if df is None:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        best = best_per_cores(df)
        cores = best["total_cores"].values
        t_serial = serial_time(df)
        speedup = t_serial / best["total_time"].values

        # ideal line
        ax.plot(cores, cores / cores[0], "--", color="#8B949E", linewidth=1.5,
                label="Ideal speedup", alpha=0.7)
        ax.plot(cores, speedup, "o-", color=color, linewidth=2.2, markersize=7,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2,
                label="Measured speedup")

        for c, s in zip(cores, speedup):
            ax.annotate(f"{s:.2f}×", xy=(c, s),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7.5, color="#C9D1D9")

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(cores)
        finish_ax(ax, label.replace("\n", " | "), "Total Cores (log₂)", "Speedup (×)")

    axes[-1].set_visible(False)
    fig.tight_layout()
    add_watermark(fig)
    fname = "plot2_speedup_per_config.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname

# ──────────────────────────────────────────────────────────────
# PLOT 3 – PARALLEL EFFICIENCY  (one per config)
# ──────────────────────────────────────────────────────────────

def plot_efficiency_per_config(datasets):
    apply_dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("Parallel Efficiency vs. Total Cores\n(E = Speedup / Cores  |  100% = perfect)",
                 fontsize=15, fontweight="bold", y=1.01)

    for idx, (df, label, color) in enumerate(zip(datasets, CONFIG_LABELS, PALETTE)):
        ax = axes[idx]
        if df is None:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        best = best_per_cores(df)
        cores = best["total_cores"].values
        t_serial = serial_time(df)
        speedup = t_serial / best["total_time"].values
        efficiency = (speedup / cores) * 100

        ax.axhline(100, linestyle="--", color="#8B949E", linewidth=1.5,
                   label="Ideal (100%)", alpha=0.7)
        ax.fill_between(cores, efficiency, 100, alpha=0.15, color=color)
        ax.plot(cores, efficiency, "o-", color=color, linewidth=2.2, markersize=7,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2,
                label="Efficiency")

        for c, e in zip(cores, efficiency):
            ax.annotate(f"{e:.1f}%", xy=(c, e),
                        xytext=(0, -14), textcoords="offset points",
                        ha="center", fontsize=7.5, color="#C9D1D9")

        ax.set_ylim(0, 130)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(cores)
        finish_ax(ax, label.replace("\n", " | "), "Total Cores (log₂)", "Efficiency (%)")

    axes[-1].set_visible(False)
    fig.tight_layout()
    add_watermark(fig)
    fname = "plot3_efficiency_per_config.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname

# ──────────────────────────────────────────────────────────────
# PLOT 4 – PHASE BREAKDOWN (stacked bar)  (one per config)
# ──────────────────────────────────────────────────────────────

def plot_phase_breakdown_per_config(datasets):
    apply_dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("Phase Breakdown vs. Total Cores\n(Interpolation | Mover | Norm | Denorm)",
                 fontsize=15, fontweight="bold", y=1.01)

    phases = ["interp_time", "norm_time", "mover_time", "denorm_time"]
    phase_names = ["Interpolation", "Normalization", "Mover", "Denormalization"]
    phase_cols = [PHASE_COLORS[p] for p in phases]

    for idx, (df, label) in enumerate(zip(datasets, CONFIG_LABELS)):
        ax = axes[idx]
        if df is None:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        best = best_per_cores(df)
        cores = best["total_cores"].values
        x = np.arange(len(cores))
        bar_w = 0.65
        bottom = np.zeros(len(cores))

        for phase, pname, pcol in zip(phases, phase_names, phase_cols):
            vals = best[phase].values
            bars = ax.bar(x, vals, bar_w, bottom=bottom, label=pname, color=pcol, alpha=0.88)
            # label bars if tall enough
            for i, (b, v) in enumerate(zip(bottom, vals)):
                if v > 0.002:
                    ax.text(x[i], b + v / 2, f"{v:.3f}",
                            ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in cores], rotation=45, fontsize=8)
        finish_ax(ax, label.replace("\n", " | "), "Total Cores", "Time (s)")

    axes[-1].set_visible(False)
    fig.tight_layout()
    add_watermark(fig)
    fname = "plot4_phase_breakdown.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname

# ──────────────────────────────────────────────────────────────
# PLOT 5 – COMBINED SPEEDUP across all configs
# ──────────────────────────────────────────────────────────────

def plot_combined_speedup(datasets):
    apply_dark_style()
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle("Combined Speedup Comparison — All Configurations",
                 fontsize=15, fontweight="bold")

    all_cores = set()
    for df in datasets:
        if df is not None:
            all_cores.update(df["total_cores"].unique())
    max_cores = max(all_cores)

    # Ideal line
    ideal_cores = np.array(sorted(all_cores))
    ax.plot(ideal_cores, ideal_cores / ideal_cores[0], "--",
            color="#8B949E", linewidth=1.5, label="Ideal", alpha=0.6)

    for df, label, color, short in zip(datasets, CONFIG_LABELS, PALETTE, CONFIG_SHORT):
        if df is None:
            continue
        best = best_per_cores(df)
        cores = best["total_cores"].values
        t_serial = serial_time(df)
        speedup = t_serial / best["total_time"].values

        ax.plot(cores, speedup, "o-", color=color, linewidth=2.4, markersize=8,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=2,
                label=short)

        # annotate max speedup
        max_idx = np.argmax(speedup)
        ax.annotate(f"max {speedup[max_idx]:.2f}×",
                    xy=(cores[max_idx], speedup[max_idx]),
                    xytext=(10, 5), textcoords="offset points",
                    fontsize=8.5, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2))

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    finish_ax(ax, "Speedup vs. Total Cores (all configs)",
              "Total Cores (log₂)", "Speedup (×)")

    fig.tight_layout()
    add_watermark(fig)
    fname = "plot5_combined_speedup.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname

# ──────────────────────────────────────────────────────────────
# PLOT 6 – HEATMAP: exec time for MPI × OMP combos
# ──────────────────────────────────────────────────────────────

def plot_heatmaps(datasets):
    apply_dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    axes = axes.flatten()
    fig.suptitle("Execution Time Heatmap: MPI Ranks × OMP Threads\n(lower = better, in seconds)",
                 fontsize=15, fontweight="bold", y=1.01)

    cmap = LinearSegmentedColormap.from_list(
        "hpc", ["#1A7F4B", "#F0E442", "#E63946"], N=256)

    for idx, (df, label) in enumerate(zip(datasets, CONFIG_LABELS)):
        ax = axes[idx]
        if df is None:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        pivot = df.pivot_table(index="mpi_ranks", columns="omp_threads",
                               values="total_time", aggfunc="min")
        mpi_vals = pivot.index.tolist()
        omp_vals = pivot.columns.tolist()
        matrix = pivot.values

        im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                       vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))

        # annotate cells
        for i in range(len(mpi_vals)):
            for j in range(len(omp_vals)):
                val = matrix[i, j]
                if not np.isnan(val):
                    # find the best cell
                    is_best = val == np.nanmin(matrix)
                    txt_color = "black" if val < (np.nanmax(matrix) * 0.6) else "white"
                    txt = f"{val:.3f}s"
                    if is_best:
                        txt += "\n★"
                    ax.text(j, i, txt, ha="center", va="center",
                            fontsize=7.5, color=txt_color, fontweight="bold" if is_best else "normal")

        ax.set_xticks(range(len(omp_vals)))
        ax.set_xticklabels([str(v) for v in omp_vals], fontsize=8)
        ax.set_yticks(range(len(mpi_vals)))
        ax.set_yticklabels([str(v) for v in mpi_vals], fontsize=8)
        ax.set_xlabel("OMP Threads", fontsize=10)
        ax.set_ylabel("MPI Ranks", fontsize=10)
        ax.set_title(label.replace("\n", " | "), fontsize=10, fontweight="bold")

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Total Time (s)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    axes[-1].set_visible(False)
    fig.tight_layout()
    add_watermark(fig)
    fname = "plot6_heatmap_mpi_omp.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname

# ──────────────────────────────────────────────────────────────
# PLOT 7 – INTERP vs MOVER phase comparison (line, per config)
# ──────────────────────────────────────────────────────────────

def plot_interp_vs_mover(datasets):
    apply_dark_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    fig.suptitle("Interpolation vs. Mover Phase Time\n(identifies the bottleneck per configuration)",
                 fontsize=15, fontweight="bold", y=1.01)

    for idx, (df, label) in enumerate(zip(datasets, CONFIG_LABELS)):
        ax = axes[idx]
        if df is None:
            ax.text(0.5, 0.5, "Data not available", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        best = best_per_cores(df)
        cores = best["total_cores"].values

        ax.plot(cores, best["interp_time"].values, "o-",
                color=PHASE_COLORS["interp_time"], linewidth=2.2, markersize=7,
                markerfacecolor="white", markeredgecolor=PHASE_COLORS["interp_time"], markeredgewidth=2,
                label="Interpolation")
        ax.plot(cores, best["mover_time"].values, "s--",
                color=PHASE_COLORS["mover_time"], linewidth=2.2, markersize=7,
                markerfacecolor="white", markeredgecolor=PHASE_COLORS["mover_time"], markeredgewidth=2,
                label="Mover")

        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.set_xticks(cores)
        finish_ax(ax, label.replace("\n", " | "), "Total Cores (log₂)", "Time (s)")

    axes[-1].set_visible(False)
    fig.tight_layout()
    add_watermark(fig)
    fname = "plot7_interp_vs_mover.png"
    fig.savefig(fname, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {fname}")
    return fname

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────

def main():
    print("\n═══════════════════════════════════════════════════════")
    print("  HPC Assignment 08 — Performance Visualization Script")
    print("═══════════════════════════════════════════════════════\n")

    datasets = []
    for f in CSV_FILES:
        df = load_data(f)
        if df is None:
            print(f"  ⚠  {f} not found — skipping (placeholder shown in plots)")
        else:
            print(f"  ✓  Loaded {f}  ({len(df)} rows, "
                  f"cores: {sorted(df['total_cores'].unique())})")
        datasets.append(df)

    print("\nGenerating plots...\n")
    saved = []
    saved.append(plot_exec_time_per_config(datasets))
    saved.append(plot_speedup_per_config(datasets))
    saved.append(plot_efficiency_per_config(datasets))
    saved.append(plot_phase_breakdown_per_config(datasets))
    saved.append(plot_combined_speedup(datasets))
    saved.append(plot_heatmaps(datasets))
    saved.append(plot_interp_vs_mover(datasets))

    print("\n═══════════════════════════════════════════════════════")
    print("  All plots saved:")
    for f in saved:
        print(f"    → {f}")
    print("\n  📌 Add timings_b.csv … timings_e.csv and re-run to")
    print("     populate the remaining configurations.")
    print("═══════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()