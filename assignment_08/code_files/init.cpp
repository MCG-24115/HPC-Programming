#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "init.h"

// Random particle initialization (optional)
void initializepoints(Points *points) {
    for (int i = 0; i < NUM_Points; i++) {
        points[i].x = (double) rand() / RAND_MAX;
        points[i].y = (double) rand() / RAND_MAX;
        points[i].is_void = false;
    }
}

// Read particle positions from binary file
void read_points(FILE *file, Points *points) 
{
    for (int i = 0; i < NUM_Points; i++) {

        if (fread(&points[i].x, sizeof(double), 1, file) != 1 ||
            fread(&points[i].y, sizeof(double), 1, file) != 1) {
            printf("Error reading input file\n");
            exit(1);
        }

        // 🔥 CRITICAL FIX
        points[i].is_void = false;
    }
}