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

    // Input files (your generated files)
    const char* input_files[] = {
        "input.bin",
        "input1.bin",
        "input2.bin",
        "input3.bin",
        "input4.bin"
    };

    const char* names[] = {"A","B","C","D","E"};

    int num_configs = 5;

    FILE *csv = fopen("serial.csv", "w");
    fprintf(csv, "Config,Threads,Time\n");

    // =========================
    // LOOP OVER CONFIG FILES
    // =========================
    for (int c = 0; c < num_configs; c++) {

        printf("\n=============================\n");
        printf("Running SERIAL Config %s\n", names[c]);
        printf("=============================\n");

        // Open binary file
        open_input_file(input_files[c]);

        int file_NX, file_NY, file_points, file_iters;
        read_input_header(&file_NX, &file_NY, &file_points, &file_iters);

        // Override parameters from file
        NX = file_NX;
        NY = file_NY;
        NUM_Points = file_points;
        Maxiter = file_iters;

        GRID_X = NX + 1;
        GRID_Y = NY + 1;

        dx = 1.0 / NX;
        dy = 1.0 / NY;

        // Allocate memory
        double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
        Points *points = (Points *) malloc(NUM_Points * sizeof(Points));

        if (!mesh_value || !points) {
            printf("Memory allocation failed\n");
            exit(1);
        }

        double total_time = 0.0;

        // =========================
        // ITERATIONS
        // =========================
        for (int iter = 0; iter < Maxiter; iter++) {

            // Read ONE iteration worth of data
            read_points_iteration(points);

            // Reset mesh
            memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

            double start = omp_get_wtime();

            interpolation_serial(mesh_value, points);

            double end = omp_get_wtime();

            total_time += (end - start);
        }

        double avg_time = total_time / Maxiter;

        printf("Avg Time: %lf\n", avg_time);

        // =========================
        // CHECKSUM (CORRECTNESS)
        // =========================
        double checksum = 0.0;
        for (int i = 0; i < GRID_X * GRID_Y; i++)
            checksum += mesh_value[i];

        printf("Checksum: %lf\n", checksum);

        // Save CSV
        fprintf(csv, "%s,1,%lf\n", names[c], avg_time);

        // Save mesh
        char filename[50];
        sprintf(filename, "mesh_serial_%s.txt", names[c]);
        save_mesh(mesh_value, filename);

        printf("Saved mesh: %s\n", filename);

        // Cleanup
        free(mesh_value);
        free(points);

        close_input_file();
    }

    fclose(csv);

    printf("\nserial.csv generated!\n");

    return 0;
}