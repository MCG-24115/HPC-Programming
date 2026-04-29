#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc,char **argv){

MPI_Init(&argc,&argv);

int rank,size;
MPI_Comm_rank(MPI_COMM_WORLD,&rank);
MPI_Comm_size(MPI_COMM_WORLD,&size);


if(argc!=2){
   if(rank==0)
      printf("Usage: %s <input_file>\n",argv[0]);
   MPI_Finalize();
   return 1;
}

FILE *file=NULL;
Points *all_points=NULL;

//----------------------------------
// Rank 0 reads metadata + particles
//----------------------------------
if(rank==0){

 file=fopen(argv[1],"rb");
 if(!file){
   printf("Error opening input file\n");
   MPI_Abort(MPI_COMM_WORLD,1);
 }

 fread(&NX,sizeof(int),1,file);
 fread(&NY,sizeof(int),1,file);
 fread(&NUM_Points,sizeof(int),1,file);
 fread(&Maxiter,sizeof(int),1,file);

 all_points=(Points*)calloc(NUM_Points,sizeof(Points));

 read_points(file,all_points);
 fclose(file);
}

//----------------------------------
// Broadcast metadata
//----------------------------------
MPI_Bcast(&NX,1,MPI_INT,0,MPI_COMM_WORLD);
MPI_Bcast(&NY,1,MPI_INT,0,MPI_COMM_WORLD);
MPI_Bcast(&NUM_Points,1,MPI_INT,0,MPI_COMM_WORLD);
MPI_Bcast(&Maxiter,1,MPI_INT,0,MPI_COMM_WORLD);

GRID_X=NX+1;
GRID_Y=NY+1;

dx=1.0/NX;
dy=1.0/NY;

//----------------------------------
// Compute particle decomposition
//----------------------------------
int *sendcounts=NULL;
int *displs=NULL;

int local_n;

int base=NUM_Points/size;
int rem=NUM_Points%size;

if(rank<rem)
   local_n=base+1;
else
   local_n=base;

if(rank==0){

 sendcounts=(int*)malloc(size*sizeof(int));
 displs=(int*)malloc(size*sizeof(int));

 int offset=0;

 for(int r=0;r<size;r++){

   int n=(r<rem)?base+1:base;

   sendcounts[r]=n*sizeof(Points);
   displs[r]=offset*sizeof(Points);

   offset += n;
 }
}

Points *points=(Points*)calloc(local_n,sizeof(Points));

//----------------------------------
// Scatter particles
//----------------------------------
if(size==1){

 memcpy(points,
        all_points,
        NUM_Points*sizeof(Points));

}
else{

MPI_Scatterv(
 all_points,
 sendcounts,
 displs,
 MPI_BYTE,
 points,
 local_n*sizeof(Points),
 MPI_BYTE,
 0,
 MPI_COMM_WORLD
);

}


if(rank==0){
 free(all_points);
 free(sendcounts);
 free(displs);
}

//----------------------------------
// Mesh allocation
//----------------------------------
int mesh_size=GRID_X*GRID_Y;

double *local_mesh=
 (double*)calloc(mesh_size,sizeof(double));

double *global_mesh=
 (double*)calloc(mesh_size,sizeof(double));

//----------------------------------
// Timers
//----------------------------------
double total_int_time=0.0;
double total_norm_time=0.0;
double total_move_time=0.0;
double total_denorm_time=0.0;


//----------------------------------
// Main iteration loop
//----------------------------------
for(int iter=0;iter<Maxiter;iter++){

 double t0=MPI_Wtime();

 interpolation(local_mesh,
               points,
               local_n);

 MPI_Allreduce(
    local_mesh,
    global_mesh,
    mesh_size,
    MPI_DOUBLE,
    MPI_SUM,
    MPI_COMM_WORLD
 );

 double t1=MPI_Wtime();

 normalization(global_mesh);

 double t2=MPI_Wtime();

 mover(global_mesh,
       points,
       local_n);

 double t3=MPI_Wtime();

 denormalization(global_mesh);

 double t4=MPI_Wtime();

 total_int_time+=(t1-t0);
 total_norm_time+=(t2-t1);
 total_move_time+=(t3-t2);
 total_denorm_time+=(t4-t3);
}

if(rank==0){
 save_mesh(global_mesh);
}

long long local_voids=
 void_count(points,local_n);

long long global_voids=0;

MPI_Reduce(
 &local_voids,
 &global_voids,
 1,
 MPI_LONG_LONG,
 MPI_SUM,
 0,
 MPI_COMM_WORLD
);

if(rank==0){

    double total_time=
        total_int_time +
        total_norm_time +
        total_move_time +
        total_denorm_time;

    printf("Total Interpolation Time = %lf seconds\n", total_int_time);
    printf("Total Normalization Time = %lf seconds\n", total_norm_time);
    printf("Total Mover Time = %lf seconds\n", total_move_time);
    printf("Total Denormalization Time = %lf seconds\n", total_denorm_time);
    printf("Total Algorithm Time = %lf seconds\n", total_time);
    printf("Total Number of Voids = %lld\n", global_voids);

    //----------------------------------
    // Append timings to CSV
    //----------------------------------
    FILE *csv=fopen("timings.csv","a");

    if(csv!=NULL){
        fprintf(csv,
        "%d,%d,%lf,%lf,%lf,%lf,%lf,%lld\n",
        size,
        omp_get_max_threads(),
        total_int_time,
        total_norm_time,
        total_move_time,
        total_denorm_time,
        total_time,
        global_voids);

        fclose(csv);
    }
}
free(local_mesh);
free(global_mesh);
free(points);

MPI_Finalize();
return 0;
}

