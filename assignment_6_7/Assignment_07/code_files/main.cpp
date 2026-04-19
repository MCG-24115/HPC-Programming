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

int main(int argc, char **argv) {

    const char* files[] = {
        "input.bin",
        "input1.bin",
        "input2.bin",
        "input3.bin",
        "input4.bin"
    };

    int thread_list[] = {2, 4, 8, 16};

    FILE *csv = fopen("results.csv", "w");
    if (!csv) {
        printf("Error opening CSV file\n");
        return 1;
    }

    fprintf(csv, "File,Threads,Interpolation,Normalization,Mover,Denormalization,Total,Voids\n");

    for (int f = 0; f < 5; f++) {

        printf("\n===== Running for %s =====\n", files[f]);

        for (int th = 0; th < 4; th++) {

            omp_set_num_threads(thread_list[th]);

            printf("\n--- Threads: %d ---\n", thread_list[th]);

            FILE *file = fopen(files[f], "rb");
            if (!file) {
                printf("Error opening input file\n");
                continue;
            }

            fread(&NX, sizeof(int), 1, file);
            fread(&NY, sizeof(int), 1, file);
            fread(&NUM_Points, sizeof(int), 1, file);
            fread(&Maxiter, sizeof(int), 1, file);

            GRID_X = NX + 1;
            GRID_Y = NY + 1;
            dx = 1.0 / NX;
            dy = 1.0 / NY;

            double *mesh_value = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
            Points *points = (Points *) calloc(NUM_Points, sizeof(Points));

            double total_int_time = 0.0;
            double total_norm_time = 0.0;
            double total_move_time = 0.0;
            double total_denorm_time = 0.0;

            read_points(file, points);

            for (int iter = 0; iter < Maxiter; iter++) {

                double t0 = omp_get_wtime();

                interpolation(mesh_value, points);

                double t1 = omp_get_wtime();

                normalization(mesh_value);

                double t2 = omp_get_wtime();

                mover(mesh_value, points);

                double t3 = omp_get_wtime();

                denormalization(mesh_value);

                double t4 = omp_get_wtime();

                total_int_time += (t1 - t0);
                total_norm_time += (t2 - t1);
                total_move_time += (t3 - t2);
                total_denorm_time += (t4 - t3);
            }

            double total_time = total_int_time + total_norm_time + total_move_time + total_denorm_time;
            long long int v = void_count(points);

            printf("Total Interpolation Time = %lf seconds\n", total_int_time);
            printf("Total Normalization Time = %lf seconds\n", total_norm_time);
            printf("Total Mover Time = %lf seconds\n", total_move_time);
            printf("Total Denormalization Time = %lf seconds\n", total_denorm_time);
            printf("Total Algorithm Time = %lf seconds\n", total_time);
            printf("Total Number of Voids = %lld\n", v);

            fprintf(csv, "%s,%d,%lf,%lf,%lf,%lf,%lf,%lld\n",
                    files[f],
                    thread_list[th],
                    total_int_time,
                    total_norm_time,
                    total_move_time,
                    total_denorm_time,
                    total_time,
                    v);

            free(mesh_value);
            free(points);
            fclose(file);
        }
    }

    fclose(csv);

    return 0;
}