#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;


// ======================
// INTERPOLATION
// ======================
void interpolation(double *mesh_value,
                   Points *points,
                   int local_n) {

    int size = GRID_X * GRID_Y;
    int nthreads = omp_get_max_threads();

    int nlocal = nthreads;

    size_t max_memory = 512 * 1024 * 1024;
    size_t per_grid = size * sizeof(double);

    if ((size_t)nthreads * per_grid > max_memory) {
        nlocal = max_memory / per_grid;
        if (nlocal < 1) nlocal = 1;
    }

    static double **local_mesh = NULL;
    static int allocated_local = 0;
    static int allocated_size = 0;

    if (local_mesh == NULL || allocated_local != nlocal || allocated_size != size) {

        if (local_mesh != NULL) {
            for (int t=0; t<allocated_local; t++)
                free(local_mesh[t]);
            free(local_mesh);
        }

        local_mesh = (double**) malloc(nlocal*sizeof(double*));

        for (int t=0; t<nlocal; t++) {
            local_mesh[t] = (double*) malloc(size*sizeof(double));
        }

        allocated_local = nlocal;
        allocated_size = size;
    }

    #pragma omp parallel for schedule(static)
    for (int t=0; t<nlocal; t++) {
        memset(local_mesh[t],0,size*sizeof(double));
    }


    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int lid = tid % nlocal;

        double *mesh_private = local_mesh[lid];

        #pragma omp for schedule(static)
        for (int p=0; p<local_n; p++) {   // CHANGED

            if (points[p].is_void) continue;

            double x=points[p].x;
            double y=points[p].y;

            int i=(int)(x/dx);
            int j=(int)(y/dy);

            if(i>=NX) i=NX-1;
            if(j>=NY) j=NY-1;

            double Xi=i*dx;
            double Yj=j*dy;

            double lx=x-Xi;
            double ly=y-Yj;

            double w00=(dx-lx)*(dy-ly);
            double w10=lx*(dy-ly);
            double w01=(dx-lx)*ly;
            double w11=lx*ly;

            int idx=j*GRID_X+i;

            mesh_private[idx]              += w00;
            mesh_private[idx+1]            += w10;
            mesh_private[idx+GRID_X]       += w01;
            mesh_private[idx+GRID_X+1]     += w11;
        }
    }


    #pragma omp parallel for schedule(static)
    for(int i=0;i<size;i++) {

        double sum=0.0;

        for(int t=0;t<nlocal;t++)
            sum += local_mesh[t][i];

        mesh_value[i]=sum;
    }
}



void normalization(double *mesh_value) {

    int size = GRID_X * GRID_Y;

    min_val=1e18;
    max_val=-1e18;

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for(int i=0;i<size;i++) {

        if(mesh_value[i] < min_val) min_val=mesh_value[i];
        if(mesh_value[i] > max_val) max_val=mesh_value[i];
    }

    double range=max_val-min_val;

    if(range==0.0) return;

    #pragma omp parallel for
    for(int i=0;i<size;i++) {

        mesh_value[i] =
            2.0*(mesh_value[i]-min_val)/range - 1.0;
    }
}



//
// MOVER
//
void mover(double *mesh_value,
           Points *points,
           int local_n) {     // CHANGED

    #pragma omp parallel for schedule(static)
    for(int p=0;p<local_n;p++) {   // CHANGED

        if(points[p].is_void) continue;

        double x=points[p].x;
        double y=points[p].y;

        int i=(int)(x/dx);
        int j=(int)(y/dy);

        if(i>=NX) i=NX-1;
        if(j>=NY) j=NY-1;

        double Xi=i*dx;
        double Yj=j*dy;

        double lx=x-Xi;
        double ly=y-Yj;

        double w00=(dx-lx)*(dy-ly);
        double w10=lx*(dy-ly);
        double w01=(dx-lx)*ly;
        double w11=lx*ly;

        int idx=j*GRID_X+i;

        double Fi =
            w00*mesh_value[idx]
          + w10*mesh_value[idx+1]
          + w01*mesh_value[idx+GRID_X]
          + w11*mesh_value[idx+GRID_X+1];

        points[p].x += Fi*dx;
        points[p].y += Fi*dy;

        if(points[p].x <0.0 || points[p].x>1.0 ||
           points[p].y <0.0 || points[p].y>1.0) {

            points[p].is_void=1;
        }
    }
}



void denormalization(double *mesh_value) {

    double range=max_val-min_val;

    if(range==0.0) return;

    int size=GRID_X*GRID_Y;

    #pragma omp parallel for
    for(int i=0;i<size;i++) {

        mesh_value[i] =
            ((mesh_value[i]+1.0)/2.0)*range + min_val;
    }
}



//
// VOID COUNT
//
long long int void_count(Points *points,
                         int local_n) {   // CHANGED

    long long int voids=0;

    #pragma omp parallel for reduction(+:voids)
    for(int i=0;i<local_n;i++) {   // CHANGED
        voids += (int)points[i].is_void;
    }

    return voids;
}



void save_mesh(double *mesh_value) {

    FILE *fd=fopen("Mesh.out","w");

    for(int i=0;i<GRID_Y;i++) {
        for(int j=0;j<GRID_X;j++) {
            fprintf(fd,"%lf ",mesh_value[i*GRID_X+j]);
        }
        fprintf(fd,"\n");
    }

    fclose(fd);
}