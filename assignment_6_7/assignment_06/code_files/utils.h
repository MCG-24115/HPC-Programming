#ifndef UTILS_H
#define UTILS_H
#include <time.h>
#include "init.h"

void interpolation(double *mesh_value, Points *points);
void interpolation_serial(double *mesh_value, Points *points);
void mover_serial(Points *points, double deltaX, double deltaY);
void mover_parallel(Points *points, double deltaX, double deltaY);
void save_mesh(double *mesh_value, const char *filename);
void open_input_file(const char *filename);
void read_input_header(int *NX, int *NY, int *NP, int *MI);
void read_points_iteration(Points *points);
void close_input_file();
#endif
