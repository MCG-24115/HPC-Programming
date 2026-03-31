/*
 * HPC Assignment 05 — Experiment 01
 * Serial Mover with Particle Deletion and Insertion
 *
 * Implements:
 *   Approach 1 — Deferred Insertion
 *   Approach 2 — Immediate Replacement
 *
 * Outputs per grid config:
 *   results_<grid>_deferred.csv   — timing, memory, FLOPs (Approach 1)
 *   results_<grid>_immediate.csv  — timing, memory, FLOPs (Approach 2)
 *   distribution_<grid>.csv       — particle x,y positions for verification
 *
 * Compile:  g++ -O2 -o particle_sim particle_sim.cpp
 * Run:      ./particle_sim
 *
 * Assignment rules:
 *   - All fractional variables use double
 *   - All integer variables use int
 *   - delta_x = 1/Nx,  delta_y = 1/Ny
 *   - Particle count stays constant throughout simulation
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <string>
#include <cstdlib>
#include <cmath>
#include <numeric>

// ================================================================
// Particle structure
// ================================================================
struct Particle {
    double x;    // position in [0, 1]
    double y;    // position in [0, 1]
    double val;  // scalar value
};

// ================================================================
// Simulation result for one (grid, np) configuration
// ================================================================
struct SimResult {
    int    np;
    int    Nx;
    int    Ny;

    // --- Timings (summed over Maxiter) ---
    double total_interp_s;
    double total_mover_s;
    double total_s;            // interp + mover

    // --- Per-particle time ---
    double time_per_particle_s;

    // --- Memory (bytes) ---
    double mem_particles_MB;
    double mem_mesh_MB;
    double mem_total_MB;

    // --- FLOPs estimates ---
    double flops_interp;       // total over all iters
    double flops_mover;        // total over all iters
    double flops_total;

    // --- Particles per cell ---
    double ppc;                // np / (Nx*Ny)

    // --- Deletion stats ---
    double avg_deletions_per_iter;
};

// ================================================================
// Timing helper
// ================================================================
double get_time_sec() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
}

// ================================================================
// RNG helpers
// ================================================================
inline double rand_double(unsigned int &seed) {
    return (double)rand_r(&seed) / (double)RAND_MAX;
}

inline double rand_displacement(double range, unsigned int &seed) {
    return range * (2.0 * rand_double(seed) - 1.0);
}

// ================================================================
// Interpolation — bilinear scatter onto Nx x Ny mesh
//
// FLOPs per particle:
//   2 div + 2 floor + 4 clamp + 2 sub + 2 div (wx,wy) = ~12
//   4 multiplies for weights + 4 add/mul for scatter = ~12
//   Total ~ 24 FLOPs/particle
// ================================================================
void interpolate(const std::vector<Particle> &particles,
                 std::vector<double> &mesh,
                 int Nx, int Ny)
{
    double dx = 1.0 / (double)Nx;
    double dy = 1.0 / (double)Ny;

    std::fill(mesh.begin(), mesh.end(), 0.0);

    int np = (int)particles.size();
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

        mesh[iy  * Nx + ix ] += w00 * particles[p].val;
        mesh[iy  * Nx + ix1] += w10 * particles[p].val;
        mesh[iy1 * Nx + ix ] += w01 * particles[p].val;
        mesh[iy1 * Nx + ix1] += w11 * particles[p].val;
    }
}

// ================================================================
// APPROACH 1 — Deferred Insertion
//
// Step 1: Move ALL particles, collect indices of those leaving domain
// Step 2: Push voids to end (tracked via index list)
// Step 3: Insert new random particles at void locations
//
// Returns number of deletions in this call
//
// FLOPs per particle:
//   2 add (displacement) + 4 compare = ~6 FLOPs
//   For deleted: 3 assignments ~ 3 FLOPs
//   Total ~ 6-9 FLOPs/particle
// ================================================================
int mover_deferred(std::vector<Particle> &particles,
                   int Nx, int Ny,
                   unsigned int &seed)
{
    double delta_x = 1.0 / (double)Nx;
    double delta_y = 1.0 / (double)Ny;

    int np = (int)particles.size();

    std::vector<int> voids;
    voids.reserve(np / 20);  // typical ~few % exit domain

    // Step 1 & 2: move all, record voids
    for (int p = 0; p < np; p++) {
        particles[p].x += rand_displacement(delta_x, seed);
        particles[p].y += rand_displacement(delta_y, seed);

        if (particles[p].x < 0.0 || particles[p].x > 1.0 ||
            particles[p].y < 0.0 || particles[p].y > 1.0)
        {
            voids.push_back(p);
        }
    }

    // Step 3: insert new particles into void slots
    int num_deleted = (int)voids.size();
    for (int i = 0; i < num_deleted; i++) {
        int idx = voids[i];
        particles[idx].x   = rand_double(seed);
        particles[idx].y   = rand_double(seed);
        particles[idx].val = rand_double(seed);
    }

    return num_deleted;
}

// ================================================================
// APPROACH 2 — Immediate Replacement
//
// For each particle:
//   Move it; if it exits domain, immediately replace at same slot
//
// Returns number of deletions in this call
// ================================================================
int mover_immediate(std::vector<Particle> &particles,
                    int Nx, int Ny,
                    unsigned int &seed)
{
    double delta_x = 1.0 / (double)Nx;
    double delta_y = 1.0 / (double)Ny;

    int np = (int)particles.size();
    int num_deleted = 0;

    for (int p = 0; p < np; p++) {
        particles[p].x += rand_displacement(delta_x, seed);
        particles[p].y += rand_displacement(delta_y, seed);

        if (particles[p].x < 0.0 || particles[p].x > 1.0 ||
            particles[p].y < 0.0 || particles[p].y > 1.0)
        {
            particles[p].x   = rand_double(seed);
            particles[p].y   = rand_double(seed);
            particles[p].val = rand_double(seed);
            num_deleted++;
        }
    }

    return num_deleted;
}

// ================================================================
// Save particle distribution (for verification plot / histogram)
// Saves x,y of first min(np, 100000) particles after last iteration
// ================================================================
void save_distribution(const std::vector<Particle> &particles,
                       const std::string &filename)
{
    std::ofstream f(filename);
    f << "x,y\n";
    int limit = std::min((int)particles.size(), 100000);
    for (int p = 0; p < limit; p++) {
        f << particles[p].x << "," << particles[p].y << "\n";
    }
    f.close();
}

// ================================================================
// Run simulation — one approach
//   approach: 1 = deferred, 2 = immediate
// ================================================================
SimResult run_simulation(int Nx, int Ny, int np, int Maxiter,
                         int approach,
                         bool save_dist,
                         const std::string &dist_file)
{
    // --- Memory calculation ---
    double mem_particles_bytes = (double)np * 3.0 * sizeof(double);
    double mem_mesh_bytes      = (double)Nx * (double)Ny * sizeof(double);
    double mem_particles_MB    = mem_particles_bytes / (1024.0 * 1024.0);
    double mem_mesh_MB         = mem_mesh_bytes      / (1024.0 * 1024.0);

    // --- FLOPs estimates ---
    // Interpolation: ~24 FLOPs per particle per iter
    double flops_interp = 24.0 * (double)np * (double)Maxiter;
    // Mover: ~8 FLOPs per particle per iter (move + check + possible replace)
    double flops_mover  =  8.0 * (double)np * (double)Maxiter;

    // --- Allocate ---
    std::vector<double>   mesh((size_t)Nx * Ny, 0.0);
    std::vector<Particle> particles(np);

    // --- One-time initialisation OUTSIDE iteration loop ---
    unsigned int seed_init = 42u;
    for (int p = 0; p < np; p++) {
        particles[p].x   = rand_double(seed_init);
        particles[p].y   = rand_double(seed_init);
        particles[p].val = rand_double(seed_init);
    }

    double total_interp = 0.0;
    double total_mover  = 0.0;
    double total_deletions = 0.0;

    unsigned int seed_mover = (approach == 1) ? 1234u : 5678u;

    // --- Iteration loop ---
    for (int iter = 0; iter < Maxiter; iter++) {
        double t0, t1;

        // Interpolation
        t0 = get_time_sec();
        interpolate(particles, mesh, Nx, Ny);
        t1 = get_time_sec();
        total_interp += (t1 - t0);

        // Mover (chosen approach)
        int deletions = 0;
        t0 = get_time_sec();
        if (approach == 1)
            deletions = mover_deferred (particles, Nx, Ny, seed_mover);
        else
            deletions = mover_immediate(particles, Nx, Ny, seed_mover);
        t1 = get_time_sec();
        total_mover += (t1 - t0);

        total_deletions += (double)deletions;
    }

    // Save distribution snapshot after last iteration
    if (save_dist)
        save_distribution(particles, dist_file);

    // --- Pack result ---
    SimResult r;
    r.np                    = np;
    r.Nx                    = Nx;
    r.Ny                    = Ny;
    r.total_interp_s        = total_interp;
    r.total_mover_s         = total_mover;
    r.total_s               = total_interp + total_mover;
    r.time_per_particle_s   = r.total_s / ((double)np * (double)Maxiter);
    r.mem_particles_MB      = mem_particles_MB;
    r.mem_mesh_MB           = mem_mesh_MB;
    r.mem_total_MB          = mem_particles_MB + mem_mesh_MB;
    r.flops_interp          = flops_interp;
    r.flops_mover           = flops_mover;
    r.flops_total           = flops_interp + flops_mover;
    r.ppc                   = (double)np / ((double)Nx * (double)Ny);
    r.avg_deletions_per_iter = total_deletions / (double)Maxiter;
    return r;
}

// ================================================================
// Write CSV header
// ================================================================
void write_csv_header(std::ofstream &f)
{
    f << "np,"
      << "Nx,"
      << "Ny,"
      << "ppc,"
      << "total_interp_s,"
      << "total_mover_s,"
      << "total_s,"
      << "time_per_particle_s,"
      << "mem_particles_MB,"
      << "mem_mesh_MB,"
      << "mem_total_MB,"
      << "flops_interp,"
      << "flops_mover,"
      << "flops_total,"
      << "avg_deletions_per_iter\n";
}

// ================================================================
// Write one CSV row
// ================================================================
void write_csv_row(std::ofstream &f, const SimResult &r)
{
    f << r.np                      << ","
      << r.Nx                      << ","
      << r.Ny                      << ","
      << r.ppc                     << ","
      << r.total_interp_s          << ","
      << r.total_mover_s           << ","
      << r.total_s                 << ","
      << r.time_per_particle_s     << ","
      << r.mem_particles_MB        << ","
      << r.mem_mesh_MB             << ","
      << r.mem_total_MB            << ","
      << r.flops_interp            << ","
      << r.flops_mover             << ","
      << r.flops_total             << ","
      << r.avg_deletions_per_iter  << "\n";
}

// ================================================================
// Main
// ================================================================
int main()
{
    const int Maxiter = 10;

    struct GridConfig {
        int Nx, Ny;
        std::string label;
    };

    std::vector<GridConfig> grids = {
        {250,  100,  "grid1_250x100" },
        {500,  200,  "grid2_500x200" },
        {1000, 400,  "grid3_1000x400"}
    };

    // -------------------------------------------------------
    // PARTICLE COUNTS
    // For Lab PC:  comment out 1000000000
    // For HPC:     keep all five
    // -------------------------------------------------------
    std::vector<int> particle_counts = {
        100,
        10000,
        1000000,
        100000000,
        // 1000000000   // <-- comment this out when running on Lab PC
    };

    for (const auto &g : grids) {

        // Output files: one per approach per grid
        std::string fname_def = "results_" + g.label + "_deferred.csv";
        std::string fname_imm = "results_" + g.label + "_immediate.csv";

        std::ofstream fout_def(fname_def);
        std::ofstream fout_imm(fname_imm);
        write_csv_header(fout_def);
        write_csv_header(fout_imm);

        std::cout << "\n=== Grid: Nx=" << g.Nx
                  << "  Ny=" << g.Ny << " ===" << std::endl;

        for (int np : particle_counts) {
            std::cout << "  np=" << np << std::flush;

            // Save distribution only for the largest np per grid
            bool save_dist = (np == particle_counts.back());
            std::string dist_file = "distribution_" + g.label + ".csv";

            // --- Approach 1: Deferred ---
            SimResult r1 = run_simulation(g.Nx, g.Ny, np, Maxiter,
                                          1, save_dist, dist_file);
            write_csv_row(fout_def, r1);
            std::cout << "  [DEF] total=" << r1.total_s << "s" << std::flush;

            // --- Approach 2: Immediate ---
            SimResult r2 = run_simulation(g.Nx, g.Ny, np, Maxiter,
                                          2, false, "");
            write_csv_row(fout_imm, r2);
            std::cout << "  [IMM] total=" << r2.total_s << "s" << std::endl;
        }

        fout_def.close();
        fout_imm.close();
        std::cout << "  Saved: " << fname_def << std::endl;
        std::cout << "  Saved: " << fname_imm << std::endl;
    }

    std::cout << "\nAll simulations complete." << std::endl;
    return 0;
}
