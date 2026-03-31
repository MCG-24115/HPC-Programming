/*
 * HPC Assignment 05 — Experiment 02
 * Parallel Mover with OpenMP: Speedup and Scalability
 *
 * Fixed: np = 14,000,000,  Maxiter = 10
 * Threads tested: 1, 2, 4, 8, 16
 *
 * Measures:
 *   - Mover WITH insertion/deletion  (Approach 1: Deferred)
 *   - Mover WITH insertion/deletion  (Approach 2: Immediate)
 *   - Mover WITHOUT insertion/deletion (plain mover, for speedup comparison)
 *   - Interpolation (parallelised)
 *
 * Outputs:
 *   results_exp02_deferred.csv
 *   results_exp02_immediate.csv
 *
 * Compile (Lab PC):
 *   g++ -O2 -std=c++11 -fopenmp -o experiment02 experiment02.cpp
 *
 * Compile (HPC — old GCC 4.8):
 *   g++ -O2 -std=c++11 -fopenmp -o experiment02 experiment02.cpp
 *
 * Run:
 *   ./experiment02
 *
 * For HPC output rename: add _hpc to output filenames (see bottom of main)
 *
 * Notes from Exp 01:
 *   - All fractional variables use double
 *   - All integer variables use int
 *   - delta_x = 1/Nx,  delta_y = 1/Ny
 *   - .c_str() used for ofstream (GCC 4.8 compatibility)
 *   - clock_gettime used instead of chrono (GCC 4.8 safe)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <omp.h>

// ================================================================
// Particle structure
// ================================================================
struct Particle {
    double x;
    double y;
    double val;
};

// ================================================================
// Timing: wall clock seconds (GCC 4.8 safe, no chrono)
// ================================================================
double get_time_sec()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

// ================================================================
// RNG helpers (thread-safe: each thread uses its own seed)
// ================================================================
inline double rand_double(unsigned int &seed)
{
    return (double)rand_r(&seed) / (double)RAND_MAX;
}

inline double rand_displacement(double range, unsigned int &seed)
{
    return range * (2.0 * rand_double(seed) - 1.0);
}

// ================================================================
// Interpolation — parallelised with OpenMP
// Bilinear scatter onto Nx x Ny mesh
// No race condition: atomic update on mesh cells
// ================================================================
void interpolate_parallel(const std::vector<Particle> &particles,
                          std::vector<double> &mesh,
                          int Nx, int Ny,
                          int num_threads)
{
    double dx = 1.0 / (double)Nx;
    double dy = 1.0 / (double)Ny;
    int np = (int)particles.size();

    std::fill(mesh.begin(), mesh.end(), 0.0);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int p = 0; p < np; p++) {
        double px = particles[p].x;
        double py = particles[p].y;

        int ix = (int)(px / dx);
        int iy = (int)(py / dy);

        if (ix >= Nx) ix = Nx - 1;
        if (iy >= Ny) iy = Ny - 1;
        if (ix < 0)   ix = 0;
        if (iy < 0)   iy = 0;

        double wx = (px - ix * dx) / dx;
        double wy = (py - iy * dy) / dy;

        double w00 = (1.0 - wx) * (1.0 - wy);
        double w10 =        wx  * (1.0 - wy);
        double w01 = (1.0 - wx) *        wy;
        double w11 =        wx  *        wy;

        int ix1 = (ix + 1 < Nx) ? ix + 1 : ix;
        int iy1 = (iy + 1 < Ny) ? iy + 1 : iy;

        #pragma omp atomic
        mesh[iy  * Nx + ix ] += w00 * particles[p].val;
        #pragma omp atomic
        mesh[iy  * Nx + ix1] += w10 * particles[p].val;
        #pragma omp atomic
        mesh[iy1 * Nx + ix ] += w01 * particles[p].val;
        #pragma omp atomic
        mesh[iy1 * Nx + ix1] += w11 * particles[p].val;
    }
}

// ================================================================
// Mover WITHOUT insertion/deletion (plain parallel mover)
// Particles reflect/clamp at boundary — same as Assignment 04
// Used for speedup comparison curve
// ================================================================
void mover_plain_parallel(std::vector<Particle> &particles,
                          int Nx, int Ny,
                          int num_threads)
{
    double delta_x = 1.0 / (double)Nx;
    double delta_y = 1.0 / (double)Ny;
    int np = (int)particles.size();

    #pragma omp parallel num_threads(num_threads)
    {
        unsigned int seed = (unsigned int)(42 + omp_get_thread_num() * 1234);

        #pragma omp for schedule(static)
        for (int p = 0; p < np; p++) {
            particles[p].x += rand_displacement(delta_x, seed);
            particles[p].y += rand_displacement(delta_y, seed);

            // Clamp to domain (no deletion)
            if (particles[p].x < 0.0) particles[p].x = 0.0;
            if (particles[p].x > 1.0) particles[p].x = 1.0;
            if (particles[p].y < 0.0) particles[p].y = 0.0;
            if (particles[p].y > 1.0) particles[p].y = 1.0;
        }
    }
}

// ================================================================
// APPROACH 1 — Deferred Insertion (Parallel)
//
// Step 1: Each thread moves its chunk, builds LOCAL void list
// Step 2: Merge void lists (serial, small cost)
// Step 3: Insert new particles at void locations (parallel)
//
// Why local void lists?
//   Avoids race condition on a shared void vector.
//   Each thread owns its segment — no locking needed.
// ================================================================
void mover_deferred_parallel(std::vector<Particle> &particles,
                              int Nx, int Ny,
                              int num_threads)
{
    double delta_x = 1.0 / (double)Nx;
    double delta_y = 1.0 / (double)Ny;
    int np = (int)particles.size();

    // Each thread gets its own void list
    std::vector< std::vector<int> > thread_voids(num_threads);

    // Step 1: parallel move + local void collection
    #pragma omp parallel num_threads(num_threads)
    {
        int tid  = omp_get_thread_num();
        unsigned int seed = (unsigned int)(1234 + tid * 9999);

        #pragma omp for schedule(static)
        for (int p = 0; p < np; p++) {
            particles[p].x += rand_displacement(delta_x, seed);
            particles[p].y += rand_displacement(delta_y, seed);

            if (particles[p].x < 0.0 || particles[p].x > 1.0 ||
                particles[p].y < 0.0 || particles[p].y > 1.0)
            {
                thread_voids[tid].push_back(p);
            }
        }
    }

    // Step 2: merge all thread void lists into one (serial, fast)
    std::vector<int> all_voids;
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < (int)thread_voids[t].size(); i++) {
            all_voids.push_back(thread_voids[t][i]);
        }
    }

    // Step 3: parallel insertion into void slots
    int num_voids = (int)all_voids.size();

    #pragma omp parallel num_threads(num_threads)
    {
        unsigned int seed = (unsigned int)(5678 + omp_get_thread_num() * 3333);

        #pragma omp for schedule(static)
        for (int i = 0; i < num_voids; i++) {
            int idx = all_voids[i];
            particles[idx].x   = rand_double(seed);
            particles[idx].y   = rand_double(seed);
            particles[idx].val = rand_double(seed);
        }
    }
}

// ================================================================
// APPROACH 2 — Immediate Replacement (Parallel)
//
// Each thread processes its own chunk independently.
// When a particle exits domain, it is replaced immediately
// at the same memory slot.
// No shared state — perfectly parallel, no race conditions.
// ================================================================
void mover_immediate_parallel(std::vector<Particle> &particles,
                               int Nx, int Ny,
                               int num_threads)
{
    double delta_x = 1.0 / (double)Nx;
    double delta_y = 1.0 / (double)Ny;
    int np = (int)particles.size();

    #pragma omp parallel num_threads(num_threads)
    {
        unsigned int seed = (unsigned int)(5678 + omp_get_thread_num() * 7777);

        #pragma omp for schedule(static)
        for (int p = 0; p < np; p++) {
            particles[p].x += rand_displacement(delta_x, seed);
            particles[p].y += rand_displacement(delta_y, seed);

            if (particles[p].x < 0.0 || particles[p].x > 1.0 ||
                particles[p].y < 0.0 || particles[p].y > 1.0)
            {
                particles[p].x   = rand_double(seed);
                particles[p].y   = rand_double(seed);
                particles[p].val = rand_double(seed);
            }
        }
    }
}

// ================================================================
// Result struct for one (grid, thread_count) run
// ================================================================
struct Exp02Result {
    int    Nx, Ny, np, num_threads;
    double t_interp;
    double t_mover_plain;     // without insertion/deletion
    double t_mover_with_del;  // with insertion/deletion
    double t_total;
    double speedup_plain;     // T_serial_plain   / T_parallel_plain
    double speedup_with_del;  // T_serial_with_del / T_parallel_with_del
};

// ================================================================
// Run Experiment 02 for one grid configuration
// approach: 1 = deferred, 2 = immediate
// ================================================================
void run_exp02(int Nx, int Ny, int np, int Maxiter,
               int approach,
               const std::string &out_filename,
               const std::string &label)
{
    std::vector<int> thread_counts;
    thread_counts.push_back(1);
    thread_counts.push_back(2);
    thread_counts.push_back(4);
    thread_counts.push_back(8);
    thread_counts.push_back(16);

    // We need serial baseline times first (1 thread)
    double serial_t_plain   = 0.0;
    double serial_t_with_del = 0.0;

    std::vector<Exp02Result> results;

    std::cout << "\n--- " << label
              << "  Approach " << approach
              << "  np=" << np << " ---" << std::endl;

    for (int ti = 0; ti < (int)thread_counts.size(); ti++) {
        int nt = thread_counts[ti];

        // Allocate mesh
        std::vector<double> mesh((size_t)Nx * Ny, 0.0);

        // Initialise particles ONCE outside iteration loop
        std::vector<Particle> p_plain(np);
        std::vector<Particle> p_with_del(np);

        unsigned int seed_init = 42u;
        for (int p = 0; p < np; p++) {
            double x   = rand_double(seed_init);
            double y   = rand_double(seed_init);
            double val = rand_double(seed_init);
            p_plain[p]    = {x, y, val};
            p_with_del[p] = {x, y, val};
        }

        double total_interp    = 0.0;
        double total_plain     = 0.0;
        double total_with_del  = 0.0;

        for (int iter = 0; iter < Maxiter; iter++) {
            double t0, t1;

            // Interpolation
            t0 = get_time_sec();
            interpolate_parallel(p_plain, mesh, Nx, Ny, nt);
            t1 = get_time_sec();
            total_interp += (t1 - t0);

            // Plain mover (no deletion) — for comparison curve
            t0 = get_time_sec();
            mover_plain_parallel(p_plain, Nx, Ny, nt);
            t1 = get_time_sec();
            total_plain += (t1 - t0);

            // Mover WITH deletion
            t0 = get_time_sec();
            if (approach == 1)
                mover_deferred_parallel (p_with_del, Nx, Ny, nt);
            else
                mover_immediate_parallel(p_with_del, Nx, Ny, nt);
            t1 = get_time_sec();
            total_with_del += (t1 - t0);
        }

        // Store serial baseline
        if (nt == 1) {
            serial_t_plain    = total_plain;
            serial_t_with_del = total_with_del;
        }

        Exp02Result r;
        r.Nx              = Nx;
        r.Ny              = Ny;
        r.np              = np;
        r.num_threads     = nt;
        r.t_interp        = total_interp;
        r.t_mover_plain   = total_plain;
        r.t_mover_with_del = total_with_del;
        r.t_total         = total_interp + total_with_del;
        r.speedup_plain   = (serial_t_plain   > 0.0) ? serial_t_plain   / total_plain    : 0.0;
        r.speedup_with_del= (serial_t_with_del > 0.0) ? serial_t_with_del / total_with_del : 0.0;

        results.push_back(r);

        std::cout << "  threads=" << nt
                  << "  plain=" << total_plain << "s"
                  << "  with_del=" << total_with_del << "s"
                  << "  speedup_plain=" << r.speedup_plain
                  << "  speedup_del=" << r.speedup_with_del
                  << std::endl;
    }

    // Write CSV
    std::ofstream fout(out_filename.c_str());
    fout << "Nx,Ny,np,num_threads,"
         << "t_interp_s,t_mover_plain_s,t_mover_with_del_s,t_total_s,"
         << "speedup_plain,speedup_with_del\n";

    for (int i = 0; i < (int)results.size(); i++) {
        Exp02Result &r = results[i];
        fout << r.Nx              << ","
             << r.Ny              << ","
             << r.np              << ","
             << r.num_threads     << ","
             << r.t_interp        << ","
             << r.t_mover_plain   << ","
             << r.t_mover_with_del<< ","
             << r.t_total         << ","
             << r.speedup_plain   << ","
             << r.speedup_with_del<< "\n";
    }
    fout.close();
    std::cout << "  Saved: " << out_filename << std::endl;
}

// ================================================================
// Main
// ================================================================
int main()
{
    const int np      = 14000000;   // fixed at 14 million
    const int Maxiter = 10;

    // -------------------------------------------------------
    // Change suffix to "_hpc" when running on HPC cluster:
    //   std::string suffix = "_hpc";
    // For Lab PC:
    std::string suffix = "";
    // -------------------------------------------------------

    struct GridConfig {
        int Nx, Ny;
        std::string label;
    };

    std::vector<GridConfig> grids;
    GridConfig g1; g1.Nx=250;  g1.Ny=100; g1.label="grid1_250x100";  grids.push_back(g1);
    GridConfig g2; g2.Nx=500;  g2.Ny=200; g2.label="grid2_500x200";  grids.push_back(g2);
    GridConfig g3; g3.Nx=1000; g3.Ny=400; g3.label="grid3_1000x400"; grids.push_back(g3);

    for (int gi = 0; gi < (int)grids.size(); gi++) {
        GridConfig &g = grids[gi];

        // Approach 1: Deferred Insertion
        std::string fname_def = "results_exp02_" + g.label + "_deferred" + suffix + ".csv";
        run_exp02(g.Nx, g.Ny, np, Maxiter, 1, fname_def, g.label);

        // Approach 2: Immediate Replacement
        std::string fname_imm = "results_exp02_" + g.label + "_immediate" + suffix + ".csv";
        run_exp02(g.Nx, g.Ny, np, Maxiter, 2, fname_imm, g.label);
    }

    std::cout << "\nExperiment 02 complete." << std::endl;
    return 0;
}
