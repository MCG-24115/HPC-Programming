"""
HPC Assignment 08 — Parallel Interpolation with Particle Mover (MPI+OpenMP)
Performance Analysis & Visualization Script
=============================================================

Plots generated (each config gets its OWN PNG file):

  Per-config (e.g. plot1_exec_time_A.png … plot1_exec_time_E.png):
    1.  Execution Time vs Total Cores
    2.  Speedup vs Total Cores  (with ideal line)
    3.  Parallel Efficiency vs Total Cores
    4.  Phase Breakdown (stacked bar)
    5.  Interpolation vs Mover phase (annotated)
    6.  Heatmap: total time for MPI × OMP combos
    7.  Amdahl's Law fit
    8.  Pure MPI vs Pure OMP vs Hybrid
    9.  Phase-level speedup (interp & mover separately)
   10.  Compute vs Overhead time

  Global (one file each):
   11.  Combined Speedup — all configs (no overlapping annotations)
   12.  Combined Execution Time — all configs
   13.  Scalability Zones — all configs

Usage:
    python plot_hpc_results.py

Place timings_a.csv … timings_e.csv in the same directory.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
CSV_FILES  = ["timings_a.csv","timings_b.csv","timings_c.csv",
              "timings_d.csv","timings_e.csv"]
CONF_LABEL = ["A: 250×100, 0.9M pts","B: 250×100, 5M pts",
              "C: 500×200, 3.6M pts","D: 500×200, 20M pts",
              "E: 1000×400, 14M pts"]
CONF_SHORT = ["A","B","C","D","E"]
PALETTE    = ["#E63946","#2196F3","#4CAF50","#FF9800","#9C27B0"]

PHASE_COL  = {"interp_time":"#E63946","mover_time":"#2196F3",
               "norm_time":"#FFC107","denorm_time":"#4CAF50"}
PHASE_NAME = {"interp_time":"Interpolation","mover_time":"Mover",
               "norm_time":"Normalization","denorm_time":"Denormalization"}

DARK = {
    "figure.facecolor":"#0D1117","axes.facecolor":"#161B22",
    "axes.edgecolor":"#30363D","axes.labelcolor":"#C9D1D9",
    "axes.titlecolor":"#F0F6FC","text.color":"#C9D1D9",
    "xtick.color":"#8B949E","ytick.color":"#8B949E",
    "grid.color":"#21262D","grid.linewidth":0.8,
    "legend.facecolor":"#21262D","legend.edgecolor":"#30363D",
    "font.family":"DejaVu Sans","font.size":11,
}

OUT = "."

# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def load(path):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["total_cores"] = df["mpi_ranks"] * df["omp_threads"]
    return df

def best_per_cores(df):
    return (df.loc[df.groupby("total_cores")["total_time"].idxmin()]
              .sort_values("total_cores").reset_index(drop=True))

def t_serial(df):
    min_c = df["total_cores"].min()
    return df[df["total_cores"]==min_c]["total_time"].min()

def DS():
    plt.rcParams.update(DARK)

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓  {name}")
    return path

def wm(fig):
    fig.text(0.99,0.01,"HPC Assignment 08 | DA-IICT",
             ha="right",va="bottom",fontsize=7.5,color="#444C56",style="italic")

def axis_style(ax, title, xlabel, ylabel, legend=True):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.minorticks_on()
    if legend:
        ax.legend(fontsize=9, framealpha=0.85, loc="best")

def log2_xticks(ax, cores):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xticks(sorted(set(cores)))

def smart_annotate(ax, xs, ys, fmt, color, fontsize=8):
    """Alternate above/below to reduce overlap."""
    offsets = [12, -18, 12, -18]
    for i,(x,y) in enumerate(zip(xs,ys)):
        dy = offsets[i % len(offsets)]
        ax.annotate(fmt(y), xy=(x,y),
                    xytext=(0,dy), textcoords="offset points",
                    ha="center", fontsize=fontsize, color=color,
                    bbox=dict(boxstyle="round,pad=0.18",fc="#161B22",
                              ec=color,lw=0.6,alpha=0.88))

def amdahl(n, f):
    return 1.0 / (f + (1-f)/n)

# ═══════════════════════════════════════════════════════════════
# PLOT 1 — Execution Time vs Total Cores
# ═══════════════════════════════════════════════════════════════
def plot_exec_time(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(9,6))
    best  = best_per_cores(df)
    cores = best["total_cores"].values
    times = best["total_time"].values
    color = PALETTE[ci]
    ax.plot(cores, times, "o-", color=color, lw=2.4, ms=8,
            mfc="white", mec=color, mew=2, label="Best MPI×OMP combo")
    smart_annotate(ax, cores, times, lambda v:f"{v:.3f}s", color)
    log2_xticks(ax, cores)
    axis_style(ax,
        f"Execution Time vs Total Cores — Config {CONF_SHORT[ci]}\n({CONF_LABEL[ci]})",
        "Total Cores (log₂ scale)", "Total Time (s)")
    wm(fig); fig.tight_layout()
    return save(fig, f"plot01_exec_time_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 2 — Speedup vs Total Cores
# ═══════════════════════════════════════════════════════════════
def plot_speedup(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(9,6))
    best  = best_per_cores(df)
    cores = best["total_cores"].values
    sp    = t_serial(df) / best["total_time"].values
    color = PALETTE[ci]
    ax.plot(cores, cores/cores[0], "--", color="#8B949E", lw=1.5,
            label="Ideal (linear)", alpha=0.7)
    ax.fill_between(cores, sp, cores/cores[0], alpha=0.08, color=color)
    ax.plot(cores, sp, "o-", color=color, lw=2.4, ms=8,
            mfc="white", mec=color, mew=2, label="Measured speedup")
    smart_annotate(ax, cores, sp, lambda v:f"{v:.2f}×", color)
    log2_xticks(ax, cores)
    axis_style(ax,
        f"Speedup vs Total Cores — Config {CONF_SHORT[ci]}\n"
        f"({CONF_LABEL[ci]})  |  S = T₁ / Tₙ",
        "Total Cores (log₂ scale)", "Speedup (×)")
    wm(fig); fig.tight_layout()
    return save(fig, f"plot02_speedup_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 3 — Parallel Efficiency
# ═══════════════════════════════════════════════════════════════
def plot_efficiency(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(9,6))
    best  = best_per_cores(df)
    cores = best["total_cores"].values
    sp    = t_serial(df) / best["total_time"].values
    eff   = (sp / cores) * 100
    color = PALETTE[ci]
    ax.axhline(100, ls="--", color="#8B949E", lw=1.5, label="Ideal (100%)", alpha=0.7)
    ax.axhspan(75,100,alpha=0.07,color="#4CAF50",label="Good (>75%)")
    ax.axhspan(50,75, alpha=0.07,color="#FFC107",label="Acceptable (50–75%)")
    ax.axhspan(0, 50, alpha=0.07,color="#E63946",label="Poor (<50%)")
    ax.fill_between(cores, eff, 100, alpha=0.12, color=color)
    ax.plot(cores, eff, "o-", color=color, lw=2.4, ms=8,
            mfc="white", mec=color, mew=2, label="Efficiency")
    smart_annotate(ax, cores, eff, lambda v:f"{v:.1f}%", color)
    ax.set_ylim(0, 130)
    log2_xticks(ax, cores)
    axis_style(ax,
        f"Parallel Efficiency vs Total Cores — Config {CONF_SHORT[ci]}\n"
        f"({CONF_LABEL[ci]})  |  E = S / N × 100",
        "Total Cores (log₂ scale)", "Efficiency (%)")
    wm(fig); fig.tight_layout()
    return save(fig, f"plot03_efficiency_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 4 — Phase Breakdown stacked bar
# ═══════════════════════════════════════════════════════════════
def plot_phase_breakdown(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(11,6))
    best   = best_per_cores(df)
    cores  = best["total_cores"].values
    x      = np.arange(len(cores))
    phases = ["interp_time","norm_time","mover_time","denorm_time"]
    bottom = np.zeros(len(cores))
    for ph in phases:
        vals = best[ph].values
        ax.bar(x, vals, 0.65, bottom=bottom,
               label=PHASE_NAME[ph], color=PHASE_COL[ph], alpha=0.88)
        for i,(b,v) in enumerate(zip(bottom,vals)):
            if v > 0.003:
                ax.text(x[i], b+v/2, f"{v:.3f}s",
                        ha="center",va="center",fontsize=7.5,
                        color="white",fontweight="bold")
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in cores], rotation=45, fontsize=9)
    axis_style(ax,
        f"Phase Breakdown vs Total Cores — Config {CONF_SHORT[ci]}\n({CONF_LABEL[ci]})",
        "Total Cores", "Time (s)")
    wm(fig); fig.tight_layout()
    return save(fig, f"plot04_phase_breakdown_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 5 — Interpolation vs Mover  FULLY ANNOTATED
# ═══════════════════════════════════════════════════════════════
def plot_interp_vs_mover(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(10,6))
    best   = best_per_cores(df)
    cores  = best["total_cores"].values
    interp = best["interp_time"].values
    mover  = best["mover_time"].values

    ax.plot(cores, interp, "o-", color=PHASE_COL["interp_time"], lw=2.4, ms=8,
            mfc="white", mec=PHASE_COL["interp_time"], mew=2, label="Interpolation (P→Grid)")
    ax.plot(cores, mover,  "s--",color=PHASE_COL["mover_time"],  lw=2.4, ms=8,
            mfc="white", mec=PHASE_COL["mover_time"],  mew=2, label="Mover (Grid→P)")

    # Interleave labels: interp above even idx, below odd; mover opposite
    for i,(x,y) in enumerate(zip(cores, interp)):
        dy = 14 if i%2==0 else -20
        ax.annotate(f"{y:.4f}s", xy=(x,y),
                    xytext=(0,dy), textcoords="offset points",
                    ha="center", fontsize=7.5, color=PHASE_COL["interp_time"],
                    bbox=dict(boxstyle="round,pad=0.18",fc="#161B22",
                              ec=PHASE_COL["interp_time"],lw=0.6,alpha=0.9))

    for i,(x,y) in enumerate(zip(cores, mover)):
        dy = -20 if i%2==0 else 14
        ax.annotate(f"{y:.4f}s", xy=(x,y),
                    xytext=(0,dy), textcoords="offset points",
                    ha="center", fontsize=7.5, color=PHASE_COL["mover_time"],
                    bbox=dict(boxstyle="round,pad=0.18",fc="#161B22",
                              ec=PHASE_COL["mover_time"],lw=0.6,alpha=0.9))

    ax.fill_between(cores, interp, mover,
                    where=(interp>mover), alpha=0.12, color=PHASE_COL["interp_time"],
                    label="Interp. dominates")
    ax.fill_between(cores, interp, mover,
                    where=(mover>=interp), alpha=0.12, color=PHASE_COL["mover_time"],
                    label="Mover dominates")

    log2_xticks(ax, cores)
    axis_style(ax,
        f"Interpolation vs Mover Phase Time — Config {CONF_SHORT[ci]}\n"
        f"({CONF_LABEL[ci]})  |  Which phase is the bottleneck?",
        "Total Cores (log₂ scale)", "Phase Time (s)")
    wm(fig); fig.tight_layout()
    return save(fig, f"plot05_interp_vs_mover_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 6 — Heatmap MPI × OMP
# ═══════════════════════════════════════════════════════════════
def plot_heatmap(df, ci):
    DS()
    pivot = df.pivot_table(index="mpi_ranks", columns="omp_threads",
                           values="total_time", aggfunc="min")
    fig, ax = plt.subplots(figsize=(10,6))
    cmap = LinearSegmentedColormap.from_list(
        "hpc",["#1A7F4B","#F0E442","#E63946"],N=256)
    mat = pivot.values
    im  = ax.imshow(mat, cmap=cmap, aspect="auto",
                    vmin=np.nanmin(mat), vmax=np.nanmax(mat))
    mpi_v = pivot.index.tolist()
    omp_v = pivot.columns.tolist()
    best_val = np.nanmin(mat)
    for i in range(len(mpi_v)):
        for j in range(len(omp_v)):
            v = mat[i,j]
            if np.isnan(v): continue
            is_best = (v == best_val)
            txt = f"{v:.3f}s\n★ Best" if is_best else f"{v:.3f}s"
            fc  = "black" if v < (np.nanmax(mat)*0.55) else "white"
            ax.text(j,i,txt,ha="center",va="center",fontsize=8.5,
                    color=fc,fontweight="bold" if is_best else "normal")
    ax.set_xticks(range(len(omp_v))); ax.set_xticklabels([str(v) for v in omp_v])
    ax.set_yticks(range(len(mpi_v))); ax.set_yticklabels([str(v) for v in mpi_v])
    ax.set_xlabel("OMP Threads",fontsize=11)
    ax.set_ylabel("MPI Ranks",  fontsize=11)
    cb = fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
    cb.set_label("Total Time (s)",fontsize=9)
    axis_style(ax,
        f"Execution Time Heatmap (MPI × OMP) — Config {CONF_SHORT[ci]}\n"
        f"({CONF_LABEL[ci]})  |  Green = fast,  Red = slow",
        "OMP Threads","MPI Ranks", legend=False)
    wm(fig); fig.tight_layout()
    return save(fig, f"plot06_heatmap_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 7 — Amdahl's Law Fit
# ═══════════════════════════════════════════════════════════════
def plot_amdahl(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(9,6))
    best  = best_per_cores(df)
    cores = best["total_cores"].values.astype(float)
    sp    = t_serial(df) / best["total_time"].values
    color = PALETTE[ci]

    ax.plot(cores, cores/cores[0], "--", color="#8B949E", lw=1.3,
            alpha=0.55, label="Ideal")
    ax.plot(cores, sp, "o", color=color, ms=9,
            mfc="white", mec=color, mew=2.5, label="Measured", zorder=5)
    smart_annotate(ax, cores, sp, lambda v:f"{v:.2f}×", color)

    try:
        popt, _ = curve_fit(amdahl, cores, sp, p0=[0.1], bounds=(0,1))
        f_serial = popt[0]
        f_par    = 1 - f_serial
        xs_fit   = np.linspace(cores.min(), cores.max()*1.5, 400)
        ax.plot(xs_fit, amdahl(xs_fit, *popt), "-",
                color=color, lw=2, alpha=0.7,
                label=f"Amdahl fit  (parallel fraction = {f_par:.1%})")
        s_max = 1/f_serial
        ax.axhline(s_max, ls=":", color="#FFC107", lw=1.5, alpha=0.8,
                   label=f"Theoretical S_max = {s_max:.1f}×")
        ax.text(cores.min()*1.1, s_max*1.02,
                f"S_max = {s_max:.1f}×  (serial fraction = {f_serial:.1%})",
                color="#FFC107", fontsize=9)
    except Exception as e:
        ax.text(0.5,0.5,f"Fit failed: {e}",transform=ax.transAxes,
                ha="center",color="#FFC107")

    log2_xticks(ax, cores)
    axis_style(ax,
        f"Amdahl's Law Fit — Config {CONF_SHORT[ci]}\n"
        f"({CONF_LABEL[ci]})  |  Identifying the serial bottleneck",
        "Total Cores (log₂ scale)", "Speedup (×)")
    wm(fig); fig.tight_layout()
    return save(fig, f"plot07_amdahl_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 8 — Pure MPI vs Pure OMP vs Hybrid
# ═══════════════════════════════════════════════════════════════
def plot_mpi_vs_omp(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(9,6))
    t0 = t_serial(df)

    pure_omp = df[df["mpi_ranks"]==1].sort_values("omp_threads")
    pure_mpi = df[df["omp_threads"]==1].sort_values("mpi_ranks")
    hybrid   = best_per_cores(df)

    all_c = []
    if not pure_omp.empty:
        c = pure_omp["omp_threads"].values
        all_c.extend(c.tolist())
        ax.plot(c, t0/pure_omp["total_time"].values, "o-",
                color="#4CAF50", lw=2.2, ms=8,
                mfc="white", mec="#4CAF50", mew=2, label="Pure OpenMP (1 MPI rank)")
    if not pure_mpi.empty:
        c = pure_mpi["mpi_ranks"].values
        all_c.extend(c.tolist())
        ax.plot(c, t0/pure_mpi["total_time"].values, "s-",
                color="#E63946", lw=2.2, ms=8,
                mfc="white", mec="#E63946", mew=2, label="Pure MPI (1 OMP thread)")
    if not hybrid.empty:
        c = hybrid["total_cores"].values
        all_c.extend(c.tolist())
        ax.plot(c, t0/hybrid["total_time"].values, "D--",
                color="#2196F3", lw=2.2, ms=8,
                mfc="white", mec="#2196F3", mew=2, label="Best Hybrid (MPI×OMP)")

    if all_c:
        all_c_s = sorted(set(all_c))
        ax.plot(all_c_s, [c/all_c_s[0] for c in all_c_s],
                "k--", lw=1.2, alpha=0.4, label="Ideal")
        log2_xticks(ax, all_c_s)

    axis_style(ax,
        f"Pure MPI vs Pure OMP vs Hybrid — Config {CONF_SHORT[ci]}\n"
        f"({CONF_LABEL[ci]})",
        "Cores / Ranks / Threads", "Speedup (×)")
    wm(fig); fig.tight_layout()
    return save(fig, f"plot08_mpi_vs_omp_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 9 — Phase-level Speedup (Interp & Mover separately)
# ═══════════════════════════════════════════════════════════════
def plot_phase_speedup(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(9,6))
    best  = best_per_cores(df)
    cores = best["total_cores"].values
    min_c = df["total_cores"].min()
    t_i0  = df[df["total_cores"]==min_c]["interp_time"].min()
    t_m0  = df[df["total_cores"]==min_c]["mover_time"].min()
    sp_i  = t_i0 / best["interp_time"].values
    sp_m  = t_m0 / best["mover_time"].values

    ax.plot(cores, cores/cores[0], "k--", lw=1.3, alpha=0.4, label="Ideal")
    ax.plot(cores, sp_i, "o-", color=PHASE_COL["interp_time"], lw=2.3, ms=8,
            mfc="white", mec=PHASE_COL["interp_time"], mew=2, label="Interpolation speedup")
    ax.plot(cores, sp_m, "s--",color=PHASE_COL["mover_time"],  lw=2.3, ms=8,
            mfc="white", mec=PHASE_COL["mover_time"],  mew=2, label="Mover speedup")

    for i,(x,y) in enumerate(zip(cores, sp_i)):
        dy = 14 if i%2==0 else -20
        ax.annotate(f"{y:.2f}×", xy=(x,y), xytext=(0,dy),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=PHASE_COL["interp_time"],
                    bbox=dict(boxstyle="round,pad=0.18",fc="#161B22",
                              ec=PHASE_COL["interp_time"],lw=0.5,alpha=0.9))
    for i,(x,y) in enumerate(zip(cores, sp_m)):
        dy = -20 if i%2==0 else 14
        ax.annotate(f"{y:.2f}×", xy=(x,y), xytext=(0,dy),
                    textcoords="offset points", ha="center", fontsize=7.5,
                    color=PHASE_COL["mover_time"],
                    bbox=dict(boxstyle="round,pad=0.18",fc="#161B22",
                              ec=PHASE_COL["mover_time"],lw=0.5,alpha=0.9))

    log2_xticks(ax, cores)
    axis_style(ax,
        f"Phase-Level Speedup — Config {CONF_SHORT[ci]}\n"
        f"({CONF_LABEL[ci]})  |  Interpolation vs Mover scaling",
        "Total Cores (log₂ scale)", "Phase Speedup (×)")
    wm(fig); fig.tight_layout()
    return save(fig, f"plot09_phase_speedup_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 10 — Compute vs Overhead time
# ═══════════════════════════════════════════════════════════════
def plot_overhead(df, ci):
    DS()
    fig, ax = plt.subplots(figsize=(9,6))
    best     = best_per_cores(df)
    cores    = best["total_cores"].values
    overhead = (best["norm_time"] + best["denorm_time"]).values
    compute  = (best["interp_time"] + best["mover_time"]).values
    color    = PALETTE[ci]

    ax.plot(cores, compute,  "s-", color=color,    lw=2.3, ms=8,
            mfc="white", mec=color,    mew=2, label="Compute (interp+mover)")
    ax.plot(cores, overhead, "o-", color="#FFC107", lw=2.3, ms=8,
            mfc="white", mec="#FFC107", mew=2, label="Sync/Norm overhead")

    ax2 = ax.twinx()
    ratio = overhead / (overhead + compute) * 100
    bar_w = np.diff(np.log2(cores.astype(float)), prepend=np.log2(cores[0])-1) * 0.25
    ax2.bar(np.log2(cores.astype(float)), ratio, width=0.25,
            color="#9C27B0", alpha=0.22, label="Overhead %")
    ax2.set_ylabel("Overhead Fraction (%)", color="#9C27B0", fontsize=10)
    ax2.tick_params(axis="y", colors="#9C27B0")
    ax2.set_ylim(0, 100)

    log2_xticks(ax, cores)
    # tweak x-axis to match log-scale for bar overlay
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())

    h1,l1 = ax.get_legend_handles_labels()
    h2,l2 = ax2.get_legend_handles_labels()
    ax.legend(h1+h2, l1+l2, fontsize=9, framealpha=0.85)
    axis_style(ax,
        f"Compute vs Overhead Time — Config {CONF_SHORT[ci]}\n"
        f"({CONF_LABEL[ci]})  |  Rising overhead kills scalability",
        "Total Cores (log₂ scale)", "Time (s)", legend=False)
    wm(fig); fig.tight_layout()
    return save(fig, f"plot10_overhead_{CONF_SHORT[ci]}.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 11 — Combined Speedup (all configs, no overlapping labels)
# ═══════════════════════════════════════════════════════════════
def plot_combined_speedup(datasets):
    DS()
    fig, ax = plt.subplots(figsize=(13,8))
    all_cores = sorted(set(c for df in datasets if df is not None
                           for c in df["total_cores"].unique()))
    base = all_cores[0]
    ax.plot(all_cores,[c/base for c in all_cores],"k--",lw=1.5,
            label="Ideal",alpha=0.45)

    # Collect max points for clean annotation
    max_pts = []   # (x, y, label, color)
    for df, ci in [(d,i) for i,d in enumerate(datasets) if d is not None]:
        best  = best_per_cores(df)
        cores = best["total_cores"].values
        sp    = t_serial(df) / best["total_time"].values
        color = PALETTE[ci]
        ax.plot(cores, sp, "o-", color=color, lw=2.2, ms=7,
                mfc="white", mec=color, mew=2,
                label=f"Config {CONF_SHORT[ci]}: {CONF_LABEL[ci]}")
        mi = int(np.argmax(sp))
        max_pts.append((cores[mi], sp[mi],
                        f"Config {CONF_SHORT[ci]}\n{sp[mi]:.2f}×", color))

    # Sort max points by y and spread them on the right margin
    max_pts.sort(key=lambda t: t[1], reverse=True)
    # assign annotation y positions with minimum gap
    ann_ys = []
    gap = 1.2
    for (xp,yp,lbl,col) in max_pts:
        ay = yp
        for prev in ann_ys:
            if abs(ay-prev) < gap:
                ay = prev - gap
        ann_ys.append(ay)
        # anchor text far right (80% of x-range on log scale)
        x_ann = all_cores[-1] * 1.08
        ax.annotate(lbl, xy=(xp,yp),
                    xytext=(x_ann, ay),
                    fontsize=8.5, color=col, fontweight="bold",
                    arrowprops=dict(arrowstyle="->",color=col,lw=1.1,
                                    connectionstyle="arc3,rad=0.0"),
                    bbox=dict(boxstyle="round,pad=0.22",fc="#161B22",
                              ec=col,lw=0.8,alpha=0.92),
                    annotation_clip=False)

    log2_xticks(ax, all_cores)
    ax.set_xlim(left=all_cores[0]*0.8)
    axis_style(ax,
        "Combined Speedup — All Configurations\nComparing scalability across problem sizes",
        "Total Cores (log₂ scale)", "Speedup (×)")
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left",
              bbox_to_anchor=(0.01,0.99))
    wm(fig); fig.tight_layout()
    return save(fig, "plot11_combined_speedup_all.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 12 — Combined Execution Time (all configs)
# ═══════════════════════════════════════════════════════════════
def plot_combined_exec_time(datasets):
    DS()
    fig, ax = plt.subplots(figsize=(12,7))
    all_cores = []
    for df, ci in [(d,i) for i,d in enumerate(datasets) if d is not None]:
        best  = best_per_cores(df)
        cores = best["total_cores"].values
        times = best["total_time"].values
        all_cores.extend(cores.tolist())
        ax.plot(cores, times, "o-", color=PALETTE[ci], lw=2.2, ms=7,
                mfc="white", mec=PALETTE[ci], mew=2,
                label=f"Config {CONF_SHORT[ci]}: {CONF_LABEL[ci]}")
    log2_xticks(ax, sorted(set(all_cores)))
    axis_style(ax,
        "Combined Execution Time — All Configurations\n"
        "Wall-clock time drop with increasing cores",
        "Total Cores (log₂ scale)", "Execution Time (s)")
    wm(fig); fig.tight_layout()
    return save(fig, "plot12_combined_exec_time_all.png")

# ═══════════════════════════════════════════════════════════════
# PLOT 13 — Scalability Zones (all configs, efficiency with bands)
# ═══════════════════════════════════════════════════════════════
def plot_scalability_zones(datasets):
    DS()
    fig, ax = plt.subplots(figsize=(12,7))
    ax.axhspan(75,100,alpha=0.07,color="#4CAF50")
    ax.axhspan(50,75, alpha=0.07,color="#FFC107")
    ax.axhspan(0, 50, alpha=0.07,color="#E63946")
    ax.axhline(100,ls="--",color="#8B949E",lw=1.3,alpha=0.6,label="Ideal (100%)")
    ax.text(1.5,103,"Ideal",fontsize=8,color="#8B949E")
    ax.text(1.5,88,"Good  (>75%)",fontsize=8,color="#4CAF50")
    ax.text(1.5,62,"Acceptable (50–75%)",fontsize=8,color="#FFC107")
    ax.text(1.5,28,"Poor  (<50%)",fontsize=8,color="#E63946")

    all_cores = []
    for df, ci in [(d,i) for i,d in enumerate(datasets) if d is not None]:
        best  = best_per_cores(df)
        cores = best["total_cores"].values
        all_cores.extend(cores.tolist())
        sp    = t_serial(df) / best["total_time"].values
        eff   = (sp / cores) * 100
        ax.plot(cores, eff, "o-", color=PALETTE[ci], lw=2.2, ms=7,
                mfc="white", mec=PALETTE[ci], mew=2,
                label=f"Config {CONF_SHORT[ci]}: {CONF_LABEL[ci]}")
    ax.set_ylim(0, 130)
    log2_xticks(ax, sorted(set(all_cores)))
    axis_style(ax,
        "Parallel Efficiency & Scalability Zones — All Configurations\n"
        "Green = good scaling  |  Yellow = acceptable  |  Red = poor",
        "Total Cores (log₂ scale)", "Parallel Efficiency (%)")
    wm(fig); fig.tight_layout()
    return save(fig, "plot13_scalability_zones_all.png")

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    sep = "═"*62
    print(f"\n{sep}")
    print("  HPC Assignment 08 — Performance Visualization Script")
    print(sep+"\n")

    datasets = []
    for f in CSV_FILES:
        df = load(f)
        if df is None:
            print(f"  ⚠  {f} not found — config will be skipped")
        else:
            print(f"  ✓  {f}  ({len(df)} rows, "
                  f"cores: {sorted(df['total_cores'].unique())})")
        datasets.append(df)

    print(f"\n{'─'*62}")
    print("  Generating per-config plots (1 PNG each)...\n")

    saved = []
    for i, df in enumerate(datasets):
        if df is None:
            print(f"  —  Config {CONF_SHORT[i]}: skipped (no data)")
            continue
        print(f"\n  Config {CONF_SHORT[i]} — {CONF_LABEL[i]}")
        saved += [
            plot_exec_time(df, i),
            plot_speedup(df, i),
            plot_efficiency(df, i),
            plot_phase_breakdown(df, i),
            plot_interp_vs_mover(df, i),
            plot_heatmap(df, i),
            plot_amdahl(df, i),
            plot_mpi_vs_omp(df, i),
            plot_phase_speedup(df, i),
            plot_overhead(df, i),
        ]

    print(f"\n{'─'*62}")
    print("  Generating global comparison plots...\n")
    saved += [
        plot_combined_speedup(datasets),
        plot_combined_exec_time(datasets),
        plot_scalability_zones(datasets),
    ]

    print(f"\n{sep}")
    print(f"  Done! {len(saved)} PNG(s) saved to: {os.path.abspath(OUT)}")
    print(sep+"\n")

if __name__ == "__main__":
    main()