#ifndef INIT_H
#define INIT_H

#include <stdio.h>

// Point structure
typedef struct {
    double x, y;
} Points;

// Global simulation parameters
extern int GRID_X, GRID_Y, NX, NY;
extern int NUM_Points, Maxiter;
extern double dx, dy;

// Initialization & I/O
void initializepoints(Points *points);
void read_points_binary(Points *points, const char *filename);
#endif