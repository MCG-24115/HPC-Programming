#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Interpolation (Serial Code)
void interpolation(double *mesh_value, Points *points) {

    int num_threads = omp_get_max_threads();

    // Allocate private grids
    double **local_grid = (double **) malloc(num_threads * sizeof(double*));

    for (int t = 0; t < num_threads; t++) {
        local_grid[t] = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    }

    // Parallel region
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *local = local_grid[tid];

        #pragma omp for schedule(static)
        for (int p = 0; p < NUM_Points; p++) {

            double x = points[p].x;
            double y = points[p].y;

            int i = (int)(x / dx);
            int j = (int)(y / dy);

            if (i >= NX) i = NX - 1;
            if (j >= NY) j = NY - 1;

            double x_i = i * dx;
            double y_j = j * dy;

            double lx = x - x_i;
            double ly = y - y_j;

            double w1 = (dx - lx) * (dy - ly);
            double w2 = ly * (dx - lx);
            double w3 = lx * (dy - ly);
            double w4 = lx * ly;

            int idx = j * GRID_X + i;

            local[idx] += w1;
            local[idx + 1] += w2;
            local[idx + GRID_X] += w3;
            local[idx + GRID_X + 1] += w4;
        }
    }

    // Reduction
    for (int t = 0; t < num_threads; t++) {
        for (int i = 0; i < GRID_X * GRID_Y; i++) {
            mesh_value[i] += local_grid[t][i];
        }
        free(local_grid[t]);
    }

    free(local_grid);
}

void interpolation_serial(double *mesh_value, Points *points) {

    for (int p = 0; p < NUM_Points; p++) {

        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x / dx);
        int j = (int)(y / dy);

        // Boundary safety
        if (i >= NX) i = NX - 1;
        if (j >= NY) j = NY - 1;

        double x_i = i * dx;
        double y_j = j * dy;

        double lx = x - x_i;
        double ly = y - y_j;

        double w1 = (dx - lx) * (dy - ly);
        double w2 = ly * (dx - lx);
        double w3 = lx * (dy - ly);
        double w4 = lx * ly;

        int idx = j * GRID_X + i;

        mesh_value[idx] += w1;
        mesh_value[idx + 1] += w2;
        mesh_value[idx + GRID_X] += w3;
        mesh_value[idx + GRID_X + 1] += w4;
    }
}

// Stochastic Mover (Serial Code) 
void mover_serial(Points *points, double deltaX, double deltaY) {}

// Stochastic Mover (Parallel Code) 
void mover_parallel(Points *points, double deltaX, double deltaY) {}

// Write mesh to file
void save_mesh(double *mesh_value, const char *filename) {

    FILE *fd = fopen(filename, "w");
    if (!fd) {
        printf("Error creating output file\n");
        exit(1);
    }

    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }

    fclose(fd);
}

FILE *fp_global = NULL;

void open_input_file(const char *filename) {
    fp_global = fopen(filename, "rb");
    if (!fp_global) {
        printf("Error opening input file %s\n", filename);
        exit(1);
    }
}

void read_input_header(int *NX_f, int *NY_f, int *NP_f, int *MI_f) {
    fread(NX_f, sizeof(int), 1, fp_global);
    fread(NY_f, sizeof(int), 1, fp_global);
    fread(NP_f, sizeof(int), 1, fp_global);
    fread(MI_f, sizeof(int), 1, fp_global);
}

void read_points_iteration(Points *points) {
    for (int i = 0; i < NUM_Points; i++) {
        fread(&points[i].x, sizeof(double), 1, fp_global);
        fread(&points[i].y, sizeof(double), 1, fp_global);
    }
}

void close_input_file() {
    fclose(fp_global);
}