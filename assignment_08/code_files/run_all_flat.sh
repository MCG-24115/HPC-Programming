# ============================================================
# CONFIG e — input4.bin (Nx=1000, Ny=400, pts=14M, Maxiter=10)
# ============================================================
echo "mpi_ranks,omp_threads,interp_time,norm_time,mover_time,denorm_time,total_time,voids" > timings_e.csv
 
# total = 2
rm -f timings.csv; export OMP_NUM_THREADS=2;  mpirun --oversubscribe --bind-to none -np 1  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=1;  mpirun --oversubscribe --bind-to none -np 2  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
 
# total = 4
rm -f timings.csv; export OMP_NUM_THREADS=4;  mpirun --oversubscribe --bind-to none -np 1  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=2;  mpirun --oversubscribe --bind-to none -np 2  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=1;  mpirun --oversubscribe --bind-to none -np 4  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
 
# total = 8
rm -f timings.csv; export OMP_NUM_THREADS=8;  mpirun --oversubscribe --bind-to none -np 1  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=4;  mpirun --oversubscribe --bind-to none -np 2  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=2;  mpirun --oversubscribe --bind-to none -np 4  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=1;  mpirun --oversubscribe --bind-to none -np 8  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
 
# total = 16
rm -f timings.csv; export OMP_NUM_THREADS=16; mpirun --oversubscribe --bind-to none -np 1  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=8;  mpirun --oversubscribe --bind-to none -np 2  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=4;  mpirun --oversubscribe --bind-to none -np 4  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=2;  mpirun --oversubscribe --bind-to none -np 8  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=1;  mpirun --oversubscribe --bind-to none -np 16 ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
 
# total = 32
rm -f timings.csv; export OMP_NUM_THREADS=32; mpirun --oversubscribe --bind-to none -np 1  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=16; mpirun --oversubscribe --bind-to none -np 2  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=8;  mpirun --oversubscribe --bind-to none -np 4  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=4;  mpirun --oversubscribe --bind-to none -np 8  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=2;  mpirun --oversubscribe --bind-to none -np 16 ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=1;  mpirun --oversubscribe --bind-to none -np 32 ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
 
# total = 64
rm -f timings.csv; export OMP_NUM_THREADS=64; mpirun --oversubscribe --bind-to none -np 1  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=32; mpirun --oversubscribe --bind-to none -np 2  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=16; mpirun --oversubscribe --bind-to none -np 4  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=8;  mpirun --oversubscribe --bind-to none -np 8  ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=4;  mpirun --oversubscribe --bind-to none -np 16 ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=2;  mpirun --oversubscribe --bind-to none -np 32 ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
rm -f timings.csv; export OMP_NUM_THREADS=1;  mpirun --oversubscribe --bind-to none -np 64 ./sim input4.bin; tail -1 timings.csv >> timings_e.csv
 
echo "Config e done → timings_e.csv"
 



echo ""
echo "ALL CONFIGS DONE."
echo "Files: timings_a.csv timings_b.csv timings_c.csv timings_d.csv timings_e.csv"
