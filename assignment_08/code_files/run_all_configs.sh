#!/bin/bash
# ============================================================
# HPC Assignment 8 — Full Run Script
# All 5 configs × all MPI×OMP pairs (total cores: 2→64)
# Input files: input.bin input1.bin input2.bin input3.bin input4.bin
# ============================================================

run_config() {
    local INPUT=$1
    local CSV=$2

    echo "mpi_ranks,omp_threads,interp_time,norm_time,mover_time,denorm_time,total_time,voids" > $CSV

    local PAIRS=(
        "1 2"   "2 1"
        "1 4"   "2 2"   "4 1"
        "1 8"   "2 4"   "4 2"   "8 1"
        "1 16"  "2 8"   "4 4"   "8 2"   "16 1"
        "1 32"  "2 16"  "4 8"   "8 4"   "16 2"  "32 1"
        "1 64"  "2 32"  "4 16"  "8 8"   "16 4"  "32 2"  "64 1"
    )

    for PAIR in "${PAIRS[@]}"; do
        RANKS=$(echo $PAIR | awk '{print $1}')
        THREADS=$(echo $PAIR | awk '{print $2}')
        echo "  Running: MPI=$RANKS  OMP=$THREADS  (total=$(( RANKS * THREADS )))"
        export OMP_NUM_THREADS=$THREADS
        rm -f timings.csv
        mpirun --oversubscribe --bind-to none -np $RANKS ./sim $INPUT
        if [ -f timings.csv ]; then
            tail -1 timings.csv >> $CSV
        fi
    done

    echo "  Done → $CSV"
}

echo "=== Config e: input4.bin (Nx=1000 Ny=400 pts=14M) ==="
run_config input4.bin timings_e.csv

echo ""
echo "All done. CSV files: timings_a.csv timings_b.csv timings_c.csv timings_d.csv timings_e.csv"
