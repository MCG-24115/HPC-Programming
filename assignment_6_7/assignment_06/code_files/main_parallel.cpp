#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main() {

    // Thread scaling
    int thread_list[] = {1, 2, 4, 8, 16};
    int num_threads = 5;

    // Input files
    const char* input_files[] = {
        "input.bin",
        "input1.bin",
        "input2.bin",
        "input3.bin",
        "input4.bin"
    };

    const char* names[] = {"A","B","C","D","E"};
    int num_configs = 5;

    FILE *csv = fopen("parallel.csv", "w");
    fprintf(csv, "Config,Threads,Time\n");

    // =========================
    // LOOP OVER CONFIGS
    // =========================
    for (int c = 0; c < num_configs; c++) {

        printf("\n=============================\n");
        printf("Running PARALLEL Config %s\n", names[c]);
        printf("=============================\n");

        // Open file ONCE to read header
        open_input_file(input_files[c]);

        int file_NX, file_NY, file_points, file_iters;
        read_input_header(&file_NX, &file_NY, &file_points, &file_iters);

        NX = file_NX;
        NY = file_NY;
        NUM_Points = file_points;
        Maxiter = file_iters;

        GRID_X = NX + 1;
        GRID_Y = NY + 1;

        dx = 1.0 / NX;
        dy = 1.0 / NY;

        close_input_file(); // close after header read

        // Allocate memory
        double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points *) malloc(NUM_Points * sizeof(Points));

        if (!mesh_value || !points) {
            printf("Memory allocation failed\n");
            exit(1);
        }

        // =========================
        // THREAD SCALING LOOP
        // =========================
        for (int t = 0; t < num_threads; t++) {

            int threads = thread_list[t];
            omp_set_num_threads(threads);

            double total_time = 0.0;

            // IMPORTANT: reopen file for each thread run
            open_input_file(input_files[c]);

            // skip header
            read_input_header(&file_NX, &file_NY, &file_points, &file_iters);

            for (int iter = 0; iter < Maxiter; iter++) {

                // read ONE iteration
                read_points_iteration(points);

                memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

                double start = omp_get_wtime();

                interpolation(mesh_value, points);

                double end = omp_get_wtime();

                total_time += (end - start);
            }

            close_input_file(); // close after reading all iterations

            double avg_time = total_time / Maxiter;

            printf("Threads: %d | Avg Time: %lf\n", threads, avg_time);

            fprintf(csv, "%s,%d,%lf\n", names[c], threads, avg_time);
        }

        // =========================
        // SAVE MESH (ONCE)
        // =========================
        omp_set_num_threads(thread_list[num_threads - 1]);

        open_input_file(input_files[c]);
        read_input_header(&file_NX, &file_NY, &file_points, &file_iters);

        // run ONE iteration for mesh output
        read_points_iteration(points);

        memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));
        interpolation(mesh_value, points);

        // checksum
        double checksum = 0.0;
        for (int i = 0; i < GRID_X * GRID_Y; i++)
            checksum += mesh_value[i];

        printf("Checksum: %lf\n", checksum);

        char filename[50];
        sprintf(filename, "mesh_parallel_%s.txt", names[c]);
        save_mesh(mesh_value, filename);

        printf("Saved mesh: %s\n", filename);

        close_input_file();

        free(mesh_value);
        free(points);
    }

    fclose(csv);

    printf("\nparallel.csv generated!\n");

    return 0;
}