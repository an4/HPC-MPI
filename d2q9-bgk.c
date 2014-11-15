/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(jj)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(ii) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   d2q9-bgk.exe input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include "mpi.h"

#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"

/* struct to hold the parameter values */
typedef struct {
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
typedef struct {
  float speeds[NSPEEDS];
} t_speed;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char* paramfile, const char* obstaclefile,
	       t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, 
	       int** obstacles_ptr, float** av_vels_ptr);

/* 
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate() & collision()
*/
int accelerate_flow(const t_param params, t_speed* cells, int* obstacles);
int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells);
int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, float* av_vels, int index);
int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
	     int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed* cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed* cells, int* obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed* cells, int* obstacles);

/* utility functions */
void die(const char* message, const int line, const char *file);
void usage(const char* exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char* argv[])
{
	char*    paramfile = NULL;    /* name of the input parameter file */
	char*    obstaclefile = NULL; /* name of a the input obstacle file */
	t_param  params;              /* struct to hold parameter values */
	t_speed* cells     = NULL;    /* grid containing fluid densities */
	t_speed* tmp_cells = NULL;    /* scratch space */
	int*     obstacles = NULL;    /* grid indicating which cells are blocked */
	float*  av_vels   = NULL;    /* a record of the av. velocity computed for each timestep */
	int      ii;                  /* generic counter */
	struct timeval timstr;        /* structure to hold elapsed time */
	struct rusage ru;             /* structure to hold CPU time--system and user */
	double tic,toc;               /* floating point numbers to calculate elapsed wallclock time */
	double usrtim;                /* floating point number to record elapsed user CPU time */
	double systim;                /* floating point number to record elapsed system CPU time */

	/* parse the command line */
	if(argc != 3) {
		usage(argv[0]);
	} else {
		paramfile = argv[1];
		obstaclefile = argv[2];
	}

	/* initialise our data structures and load values from file */
	initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels);

	// Initialize MPI environment.
	MPI_Init(&argc, &argv);

	int flag;
	// Check if initialization was successful.
	MPI_Initialized(&flag);
	if(flag != 1) {
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	int rank,size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	/* iterate for maxIters timesteps */
	gettimeofday(&timstr,NULL);
	tic=timstr.tv_sec+(timstr.tv_usec/1000000.0);

	for(ii=0;ii<params.maxIters;ii++) {
		accelerate_flow(params,cells,obstacles);
		propagate(params,cells,tmp_cells);
		collision(params,cells,tmp_cells,obstacles, av_vels, ii);	
		

		#ifdef DEBUG
		printf("==timestep: %d==\n",ii);
		printf("av velocity: %.12E\n", av_vels[ii]);
		printf("tot density: %.12E\n",total_density(params,cells));
		#endif
	}

	gettimeofday(&timstr,NULL);
	toc=timstr.tv_sec+(timstr.tv_usec/1000000.0);
	getrusage(RUSAGE_SELF, &ru);
	timstr=ru.ru_utime;        
	usrtim=timstr.tv_sec+(timstr.tv_usec/1000000.0);
	timstr=ru.ru_stime;        
	systim=timstr.tv_sec+(timstr.tv_usec/1000000.0);

	// Finalize MPI environment.
	MPI_Finalize();

	MPI_Finalized(&flag);
	if(flag != 1) {
		MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
	}

	/* write final values and free memory */
	printf("==done==\n");
	printf("Reynolds number:\t\t%.12E\n",calc_reynolds(params,cells,obstacles));
	printf("Elapsed time:\t\t\t%.6lf (s)\n", toc-tic);
	printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
	printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
	write_values(params,cells,obstacles,av_vels);
	finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

	return EXIT_SUCCESS;
}

int accelerate_flow(const t_param params, t_speed* cells, int* obstacles)
{
  int ii;     /* generic counters */
  
  /* compute weighting factors */
  float w2 = params.density * params.accel * 0.02777777777;
  float w1 = w2 * 4;
  
  /* modify the 2nd row of the grid */
  for(ii=(params.ny-2)*params.nx;ii<(params.ny-1)*params.nx;ii++) {
    /* if the cell is not occupied and
    ** we don't send a density negative */
    if( !obstacles[ii] && 
	(cells[ii].speeds[3] - w1) > 0.0 &&
	(cells[ii].speeds[6] - w2) > 0.0 &&
	(cells[ii].speeds[7] - w2) > 0.0 ) {
      /* increase 'east-side' densities */
      cells[ii].speeds[1] += w1;
      cells[ii].speeds[5] += w2;
      cells[ii].speeds[8] += w2;
      /* decrease 'west-side' densities */
      cells[ii].speeds[3] -= w1;
      cells[ii].speeds[6] -= w2;
      cells[ii].speeds[7] -= w2;
    }
  }

  return EXIT_SUCCESS;
}

int propagate(const t_param params, t_speed* cells, t_speed* tmp_cells)
{
  int ii;
  
  // Lower left corner
  tmp_cells[0].speeds[0] = cells[0].speeds[0];
  tmp_cells[0].speeds[1] = cells[params.nx-1].speeds[1];
  tmp_cells[0].speeds[2] = cells[params.nx*(params.ny-1)].speeds[2];
  tmp_cells[0].speeds[3] = cells[1].speeds[3];
  tmp_cells[0].speeds[4] = cells[params.nx].speeds[4];
  tmp_cells[0].speeds[5] = cells[params.nx*params.ny-1].speeds[5];
  tmp_cells[0].speeds[6] = cells[params.nx*(params.ny-1)+1].speeds[6];
  tmp_cells[0].speeds[7] = cells[params.nx+1].speeds[7];
  tmp_cells[0].speeds[8] = cells[2*params.nx-1].speeds[8];
  
  // Lower right corner
  tmp_cells[params.nx-1].speeds[0] = cells[params.nx-1].speeds[0];
  tmp_cells[params.nx-1].speeds[1] = cells[params.nx-2].speeds[1];
  tmp_cells[params.nx-1].speeds[2] = cells[(params.nx-1)*(params.ny-1)].speeds[2];
  tmp_cells[params.nx-1].speeds[3] = cells[0].speeds[3];
  tmp_cells[params.nx-1].speeds[4] = cells[2*params.nx-1].speeds[4];
  tmp_cells[params.nx-1].speeds[5] = cells[params.nx*params.ny-2].speeds[5];
  tmp_cells[params.nx-1].speeds[6] = cells[params.nx*(params.ny-1)].speeds[6];
  tmp_cells[params.nx-1].speeds[7] = cells[params.nx].speeds[7];
  tmp_cells[params.nx-1].speeds[8] = cells[2*params.nx-2].speeds[8];
  
  // Upper left corner
  tmp_cells[params.nx*(params.ny-1)].speeds[0] = cells[params.nx*(params.ny-1)].speeds[0];
  tmp_cells[params.nx*(params.ny-1)].speeds[1] = cells[params.nx*params.ny-1].speeds[1];
  tmp_cells[params.nx*(params.ny-1)].speeds[2] = cells[params.nx*(params.ny-2)].speeds[2];
  tmp_cells[params.nx*(params.ny-1)].speeds[3] = cells[params.nx*(params.ny-1)+1].speeds[3];
  tmp_cells[params.nx*(params.ny-1)].speeds[4] = cells[0].speeds[4];
  tmp_cells[params.nx*(params.ny-1)].speeds[5] = cells[params.nx*(params.ny-1)-1].speeds[5];
  tmp_cells[params.nx*(params.ny-1)].speeds[6] = cells[params.nx*(params.ny-2)+1].speeds[6];
  tmp_cells[params.nx*(params.ny-1)].speeds[7] = cells[1].speeds[7];
  //tmp_cells[params.nx*(params.ny-1)].speeds[8] = cells[params.nx-1].speeds[8];
  
  // Upper right corner
  tmp_cells[params.nx*params.ny-1].speeds[0] = cells[params.nx*params.ny-1].speeds[0];
  tmp_cells[params.nx*params.ny-1].speeds[1] = cells[params.nx*params.ny-2].speeds[1];
  tmp_cells[params.nx*params.ny-1].speeds[2] = cells[params.nx*(params.ny-1)-1].speeds[2];
  tmp_cells[params.nx*params.ny-1].speeds[3] = cells[params.nx*(params.ny-1)].speeds[3];
  tmp_cells[params.nx*params.ny-1].speeds[4] = cells[params.nx-1].speeds[4];
  tmp_cells[params.nx*params.ny-1].speeds[5] = cells[params.nx*(params.ny-1)-2].speeds[5];
  tmp_cells[params.nx*params.ny-1].speeds[6] = cells[params.nx*(params.ny-2)].speeds[6];
  tmp_cells[params.nx*params.ny-1].speeds[7] = cells[0].speeds[7];
  tmp_cells[params.nx*params.ny-1].speeds[8] = cells[params.nx-2].speeds[8];
  
  // First row
  for(ii=1;ii<params.ny-1;ii++) {
    tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
  	tmp_cells[ii].speeds[1] = cells[ii-1].speeds[1];
    tmp_cells[ii].speeds[2] = cells[params.nx*(params.ny-1)+ii].speeds[2];
    tmp_cells[ii].speeds[3] = cells[ii+1].speeds[3];
    tmp_cells[ii].speeds[4] = cells[ii+params.nx].speeds[4];
    tmp_cells[ii].speeds[5] = cells[params.nx*(params.ny-1)+ii-1].speeds[5];
    tmp_cells[ii].speeds[6] = cells[params.nx*(params.ny-1)+ii+1].speeds[6];
    tmp_cells[ii].speeds[7] = cells[params.nx+ii+1].speeds[7];
    tmp_cells[ii].speeds[8] = cells[params.nx+ii-1].speeds[8];
  }
  
  // Last row
  for(ii=params.nx*(params.ny-1)+1;ii<params.ny*params.nx-1;ii++) {
    tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
  	tmp_cells[ii].speeds[1] = cells[ii-1].speeds[1];
    tmp_cells[ii].speeds[2] = cells[ii-params.nx].speeds[2];
    tmp_cells[ii].speeds[3] = cells[ii+1].speeds[3];
    tmp_cells[ii].speeds[4] = cells[ii-params.nx*(params.ny-1)].speeds[4];
    tmp_cells[ii].speeds[5] = cells[ii-params.nx-1].speeds[5];
    tmp_cells[ii].speeds[6] = cells[ii-params.nx+1].speeds[6];
    tmp_cells[ii].speeds[7] = cells[ii-params.nx*(params.ny-1)+1].speeds[7];
    tmp_cells[ii].speeds[8] = cells[ii-params.nx*(params.ny-1)-1].speeds[8];
  }
  
  for(ii=params.nx; ii<params.nx*(params.ny-1); ii++) {
  	// First column
  	if(ii%params.nx==0) {
  	  tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
  	  tmp_cells[ii].speeds[1] = cells[ii+params.nx-1].speeds[1];
      tmp_cells[ii].speeds[2] = cells[ii-params.nx].speeds[2];
      tmp_cells[ii].speeds[3] = cells[ii+1].speeds[3];
      tmp_cells[ii].speeds[4] = cells[ii+params.nx].speeds[4];
      tmp_cells[ii].speeds[5] = cells[ii-1].speeds[5];
      tmp_cells[ii].speeds[6] = cells[ii-params.nx+1].speeds[6];
      tmp_cells[ii].speeds[7] = cells[ii+params.nx+1].speeds[7];
      tmp_cells[ii].speeds[8] = cells[ii+2*params.nx-1].speeds[8];
      continue;
  	}
  	// Last column
  	if((ii+1)%params.nx==0) {
  	  tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
  	  tmp_cells[ii].speeds[1] = cells[ii-1].speeds[1];
      tmp_cells[ii].speeds[2] = cells[ii-params.nx].speeds[2];
      tmp_cells[ii].speeds[3] = cells[ii-params.nx+1].speeds[3];
      tmp_cells[ii].speeds[4] = cells[ii+params.nx].speeds[4];
      tmp_cells[ii].speeds[5] = cells[ii-params.nx-1].speeds[5];
      tmp_cells[ii].speeds[6] = cells[ii-2*params.nx+1].speeds[6];
      tmp_cells[ii].speeds[7] = cells[ii+1].speeds[7];
      tmp_cells[ii].speeds[8] = cells[ii+params.nx-1].speeds[8];
      continue;
  	}
  	// General
  	tmp_cells[ii].speeds[0] = cells[ii].speeds[0];
  	tmp_cells[ii].speeds[1] = cells[ii-1].speeds[1];
    tmp_cells[ii].speeds[2] = cells[ii-params.nx].speeds[2];
    tmp_cells[ii].speeds[3] = cells[ii+1].speeds[3];
    tmp_cells[ii].speeds[4] = cells[ii+params.nx].speeds[4];
    tmp_cells[ii].speeds[5] = cells[ii-params.nx-1].speeds[5];
    tmp_cells[ii].speeds[6] = cells[ii-params.nx+1].speeds[6];
    tmp_cells[ii].speeds[7] = cells[ii+params.nx+1].speeds[7];
    tmp_cells[ii].speeds[8] = cells[ii+params.nx-1].speeds[8];
  }

  return EXIT_SUCCESS;
}

int collision(const t_param params, t_speed* cells, t_speed* tmp_cells, int* obstacles, float* av_vels, int index)
{
	int ii,kk;                    /* generic counters */
	float u[NSPEEDS];            /* directional velocities */
	float d_equ[NSPEEDS];        /* equilibrium densities */
	int    tot_cells = 0;         /* no. of cells used in calculation */
	float tot_u = 0.0;           /* accumulated magnitudes of velocity for each cell */

	// MPI variables.
	int size;
	int rank;
	MPI_Status status;  

	// Get number of processes.
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// The number of cells assigned to each process;
	int piece = (params.nx*params.ny)/size;

	// Master = size-1

	// Determine rank of current process
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	
	//for(ii=0;ii<params.ny*params.nx;ii++) {
	for(ii=rank*piece; ii<(rank+1)*piece; ii++) {
		/* don't consider occupied cells */
		if(!obstacles[ii]) {
			/* compute local density total */
			float local_density = 0.0;
			for(kk=0;kk<NSPEEDS;kk++) {
				local_density += tmp_cells[ii].speeds[kk];
			}
			/* compute x velocity component */
			float u_x = (tmp_cells[ii].speeds[1] + 
			       tmp_cells[ii].speeds[5] + 
			       tmp_cells[ii].speeds[8]
			       - (tmp_cells[ii].speeds[3] + 
			      tmp_cells[ii].speeds[6] + 
			      tmp_cells[ii].speeds[7]))
			  / local_density;
			/* compute y velocity component */
			float u_y = (tmp_cells[ii].speeds[2] + 
			       tmp_cells[ii].speeds[5] + 
			       tmp_cells[ii].speeds[6]
			       - (tmp_cells[ii].speeds[4] + 
			      tmp_cells[ii].speeds[7] + 
			      tmp_cells[ii].speeds[8]))
			  / local_density;
			  
			/* accumulate the norm of x- and y- velocity components */
			tot_u += sqrt((u_x * u_x) + (u_y * u_y));
			/* increase counter of inspected cells */
			++tot_cells;

			/* velocity squared */ 
			float u_sq = u_x * u_x + u_y * u_y;
			/* directional velocity components */
			u[1] =   u_x;        /* east */
			u[2] =         u_y;  /* north */
			u[3] = - u_x;        /* west */
			u[4] =       - u_y;  /* south */
			u[5] =   u_x + u_y;  /* north-east */
			u[6] = - u_x + u_y;  /* north-west */
			u[7] = - u_x - u_y;  /* south-west */
			u[8] =   u_x - u_y;  /* south-east */
			/* equilibrium densities */

			// 1/(2.0 * c_sq * c_sq) = 4.5
			// 1/c_sq = 3.0
			// 1/(2.0 * c_sq) = 1.5
			// w0 = 0.44444444444
			// w1 = 0.11111111111
			// w2 = 0.02777777777

			/* zero velocity density: weight w0 */
			d_equ[0] = 0.44444444444 * local_density * (1.0 - u_sq * 1.5);
			/* axis speeds: weight w1 */
			d_equ[1] = 0.11111111111 * local_density * (1.0 + u[1] * 3.0
				             + (u[1] * u[1]) * 4.5
				             - u_sq * 1.5);
			d_equ[2] = 0.11111111111 * local_density * (1.0 + u[2] * 3.0
				             + (u[2] * u[2]) * 4.5
				             - u_sq * 1.5);
			d_equ[3] = 0.11111111111 * local_density * (1.0 + u[3] * 3.0
				             + (u[3] * u[3]) * 4.5
				             - u_sq * 1.5);
			d_equ[4] = 0.11111111111 * local_density * (1.0 + u[4] * 3.0
				             + (u[4] * u[4]) * 4.5
				             - u_sq * 1.5);
			/* diagonal speeds: weight w2 */
			d_equ[5] = 0.02777777777 * local_density * (1.0 + u[5] * 3.0
				             + (u[5] * u[5]) * 4.5
				             - u_sq * 1.5);
			d_equ[6] = 0.02777777777 * local_density * (1.0 + u[6] * 3.0
				             + (u[6] * u[6]) * 4.5
				             - u_sq * 1.5);
			d_equ[7] = 0.02777777777 * local_density * (1.0 + u[7] * 3.0
				             + (u[7] * u[7]) * 4.5
				             - u_sq * 1.5);
			d_equ[8] = 0.02777777777 * local_density * (1.0 + u[8] * 3.0
				             + (u[8] * u[8]) * 4.5
				             - u_sq * 1.5);
			/* relaxation step */
			for(kk=0;kk<NSPEEDS;kk++) {
				cells[ii].speeds[kk] = (tmp_cells[ii].speeds[kk]
					             + params.omega * 
					             (d_equ[kk] - tmp_cells[ii].speeds[kk]));
			}
		} else {
			/* called after propagate, so taking values from scratch space
			** mirroring, and writing into main grid */
			cells[ii].speeds[1] = tmp_cells[ii].speeds[3];
			cells[ii].speeds[2] = tmp_cells[ii].speeds[4];
			cells[ii].speeds[3] = tmp_cells[ii].speeds[1];
			cells[ii].speeds[4] = tmp_cells[ii].speeds[2];
			cells[ii].speeds[5] = tmp_cells[ii].speeds[7];
			cells[ii].speeds[6] = tmp_cells[ii].speeds[8];
			cells[ii].speeds[7] = tmp_cells[ii].speeds[5];
			cells[ii].speeds[8] = tmp_cells[ii].speeds[6];
		}
	}

	int total_cells = 0;
	MPI_Reduce(&tot_cells, &total_cells, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank == 0) {

		float total_u = tot_u;
		int i;
		for(i=1; i<size;i++) {
			float temp_u = 0.0;
			MPI_Recv(&temp_u, 1, MPI_FLOAT, i, index, MPI_COMM_WORLD);
		}

		printf("I:%d\t\t%f %d\n",index+1,total_u,total_cells);
		av_vels[index] = total_u / (float)total_cells;
		return EXIT_SUCCESS;
	} else {
		MPI_Send(&tot_u, 1, MPI_FLOAT, 0, index, MPI_COMM_WORLD);
	}
}

int initialise(const char* paramfile, const char* obstaclefile,
	       t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr, 
	       int** obstacles_ptr, float** av_vels_ptr)
{
  char   message[1024];  /* message buffer */
  FILE   *fp;            /* file pointer */
  int    ii,jj;          /* generic counters */
  int    xx,yy;          /* generic array indices */
  int    blocked;        /* indicates whether a cell is blocked by an obstacle */ 
  int    retval;         /* to hold return value for checking */
  float w0,w1,w2;       /* weighting factors */

  /* open the parameter file */
  fp = fopen(paramfile,"r");
  if (fp == NULL) {
    sprintf(message,"could not open input parameter file: %s", paramfile);
    die(message,__LINE__,__FILE__);
  }

  /* read in the parameter values */
  retval = fscanf(fp,"%d\n",&(params->nx));
  if(retval != 1) die ("could not read param file: nx",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->ny));
  if(retval != 1) die ("could not read param file: ny",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->maxIters));
  if(retval != 1) die ("could not read param file: maxIters",__LINE__,__FILE__);
  retval = fscanf(fp,"%d\n",&(params->reynolds_dim));
  if(retval != 1) die ("could not read param file: reynolds_dim",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->density));
  if(retval != 1) die ("could not read param file: density",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->accel));
  if(retval != 1) die ("could not read param file: accel",__LINE__,__FILE__);
  retval = fscanf(fp,"%f\n",&(params->omega));
  if(retval != 1) die ("could not read param file: omega",__LINE__,__FILE__);

  /* and close up the file */
  fclose(fp);

  /* 
  ** Allocate memory.
  **
  ** Remember C is pass-by-value, so we need to
  ** pass pointers into the initialise function.
  **
  ** NB we are allocating a 1D array, so that the
  ** memory will be contiguous.  We still want to
  ** index this memory as if it were a (row major
  ** ordered) 2D array, however.  We will perform
  ** some arithmetic using the row and column
  ** coordinates, inside the square brackets, when
  ** we want to access elements of this array.
  **
  ** Note also that we are using a structure to
  ** hold an array of 'speeds'.  We will allocate
  ** a 1D array of these structs.
  */

  /* main grid */
  *cells_ptr = (t_speed*)malloc(sizeof(t_speed)*(params->ny*params->nx));
  if (*cells_ptr == NULL) 
    die("cannot allocate memory for cells",__LINE__,__FILE__);

  /* 'helper' grid, used as scratch space */
  *tmp_cells_ptr = (t_speed*)malloc(sizeof(t_speed)*(params->ny*params->nx));
  if (*tmp_cells_ptr == NULL) 
    die("cannot allocate memory for tmp_cells",__LINE__,__FILE__);
  
  /* the map of obstacles */
  *obstacles_ptr = malloc(sizeof(int*)*(params->ny*params->nx));
  if (*obstacles_ptr == NULL) 
    die("cannot allocate column memory for obstacles",__LINE__,__FILE__);

  /* initialise densities */
  w0 = params->density * 4.0/9.0;
  w1 = params->density      /9.0;
  w2 = params->density      /36.0;

  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      /* centre */
      (*cells_ptr)[ii*params->nx + jj].speeds[0] = w0;
      /* axis directions */
      (*cells_ptr)[ii*params->nx + jj].speeds[1] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[2] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[3] = w1;
      (*cells_ptr)[ii*params->nx + jj].speeds[4] = w1;
      /* diagonals */
      (*cells_ptr)[ii*params->nx + jj].speeds[5] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[6] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[7] = w2;
      (*cells_ptr)[ii*params->nx + jj].speeds[8] = w2;
    }
  }

  /* first set all cells in obstacle array to zero */ 
  for(ii=0;ii<params->ny;ii++) {
    for(jj=0;jj<params->nx;jj++) {
      (*obstacles_ptr)[ii*params->nx + jj] = 0;
    }
  }

  /* open the obstacle data file */
  fp = fopen(obstaclefile,"r");
  if (fp == NULL) {
    sprintf(message,"could not open input obstacles file: %s", obstaclefile);
    die(message,__LINE__,__FILE__);
  }

  /* read-in the blocked cells list */
  while( (retval = fscanf(fp,"%d %d %d\n", &xx, &yy, &blocked)) != EOF) {
    /* some checks */
    if ( retval != 3)
      die("expected 3 values per line in obstacle file",__LINE__,__FILE__);
    if ( xx<0 || xx>params->nx-1 )
      die("obstacle x-coord out of range",__LINE__,__FILE__);
    if ( yy<0 || yy>params->ny-1 )
      die("obstacle y-coord out of range",__LINE__,__FILE__);
    if ( blocked != 1 ) 
      die("obstacle blocked value should be 1",__LINE__,__FILE__);
    /* assign to array */
    (*obstacles_ptr)[yy*params->nx + xx] = blocked;
  }
  
  /* and close the file */
  fclose(fp);

  /* 
  ** allocate space to hold a record of the avarage velocities computed 
  ** at each timestep
  */
  *av_vels_ptr = (float*)malloc(sizeof(float)*params->maxIters);

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, t_speed** cells_ptr, t_speed** tmp_cells_ptr,
	     int** obstacles_ptr, float** av_vels_ptr)
{
  /* 
  ** free up allocated memory
  */
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*tmp_cells_ptr);
  *tmp_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed* cells, int* obstacles)
{
  int    ii,jj,kk;       /* generic counters */
  int    tot_cells = 0;  /* no. of cells used in calculation */
  float local_density;  /* total density in cell */
  float u_x;            /* x-component of velocity for current cell */
  float u_y;            /* y-component of velocity for current cell */
  float tot_u;          /* accumulated magnitudes of velocity for each cell */

  /* initialise */
  tot_u = 0.0;

  /* loop over all non-blocked cells */
  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* ignore occupied cells */
      if(!obstacles[ii*params.nx + jj]) {
	/* local density total */
	local_density = 0.0;
	for(kk=0;kk<NSPEEDS;kk++) {
	  local_density += cells[ii*params.nx + jj].speeds[kk];
	}
	/* x-component of velocity */
	u_x = (cells[ii*params.nx + jj].speeds[1] + 
		    cells[ii*params.nx + jj].speeds[5] + 
		    cells[ii*params.nx + jj].speeds[8]
		    - (cells[ii*params.nx + jj].speeds[3] + 
		       cells[ii*params.nx + jj].speeds[6] + 
		       cells[ii*params.nx + jj].speeds[7])) / 
	  local_density;
	/* compute y velocity component */
	u_y = (cells[ii*params.nx + jj].speeds[2] + 
		    cells[ii*params.nx + jj].speeds[5] + 
		    cells[ii*params.nx + jj].speeds[6]
		    - (cells[ii*params.nx + jj].speeds[4] + 
		       cells[ii*params.nx + jj].speeds[7] + 
		       cells[ii*params.nx + jj].speeds[8])) /
	  local_density;
	/* accumulate the norm of x- and y- velocity components */
	tot_u += sqrt((u_x * u_x) + (u_y * u_y));
	/* increase counter of inspected cells */
	++tot_cells;
      }
    }
  }

  return tot_u / (float)tot_cells;
}

float calc_reynolds(const t_param params, t_speed* cells, int* obstacles)
{
  const float viscosity = 1.0 / 6.0 * (2.0 / params.omega - 1.0);
  
  return av_velocity(params,cells,obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed* cells)
{
  int ii,jj,kk;        /* generic counters */
  float total = 0.0;  /* accumulator */

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      for(kk=0;kk<NSPEEDS;kk++) {
	total += cells[ii*params.nx + jj].speeds[kk];
      }
    }
  }
  
  return total;
}

int write_values(const t_param params, t_speed* cells, int* obstacles, float* av_vels)
{
  FILE* fp;                     /* file pointer */
  int ii,jj,kk;                 /* generic counters */
  const float c_sq = 1.0/3.0;  /* sq. of speed of sound */
  float local_density;         /* per grid cell sum of densities */
  float pressure;              /* fluid pressure in grid cell */
  float u_x;                   /* x-component of velocity in grid cell */
  float u_y;                   /* y-component of velocity in grid cell */
  float u;                     /* norm--root of summed squares--of u_x and u_y */

  fp = fopen(FINALSTATEFILE,"w");
  if (fp == NULL) {
    die("could not open file output file",__LINE__,__FILE__);
  }

  for(ii=0;ii<params.ny;ii++) {
    for(jj=0;jj<params.nx;jj++) {
      /* an occupied cell */
      if(obstacles[ii*params.nx + jj]) {
	u_x = u_y = u = 0.0;
	pressure = params.density * c_sq;
      }
      /* no obstacle */
      else {
	local_density = 0.0;
	for(kk=0;kk<NSPEEDS;kk++) {
	  local_density += cells[ii*params.nx + jj].speeds[kk];
	}
	/* compute x velocity component */
	u_x = (cells[ii*params.nx + jj].speeds[1] + 
	       cells[ii*params.nx + jj].speeds[5] +
	       cells[ii*params.nx + jj].speeds[8]
	       - (cells[ii*params.nx + jj].speeds[3] + 
		  cells[ii*params.nx + jj].speeds[6] + 
		  cells[ii*params.nx + jj].speeds[7]))
	  / local_density;
	/* compute y velocity component */
	u_y = (cells[ii*params.nx + jj].speeds[2] + 
	       cells[ii*params.nx + jj].speeds[5] + 
	       cells[ii*params.nx + jj].speeds[6]
	       - (cells[ii*params.nx + jj].speeds[4] + 
		  cells[ii*params.nx + jj].speeds[7] + 
		  cells[ii*params.nx + jj].speeds[8]))
	  / local_density;
	/* compute norm of velocity */
	u = sqrt((u_x * u_x) + (u_y * u_y));
	/* compute pressure */
	pressure = local_density * c_sq;
      }
      /* write to file */
      fprintf(fp,"%d %d %.12E %.12E %.12E %.12E %d\n",jj,ii,u_x,u_y,u,pressure,obstacles[ii*params.nx + jj]);
    }
  }

  fclose(fp);

  fp = fopen(AVVELSFILE,"w");
  if (fp == NULL) {
    die("could not open file output file",__LINE__,__FILE__);
  }
  for (ii=0;ii<params.maxIters;ii++) {
    fprintf(fp,"%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);

  return EXIT_SUCCESS;
}

void die(const char* message, const int line, const char *file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n",message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}
