#include "streamcluster_header.cu"

using namespace std;

#define CUDA_SAFE_CALL( call) do {										\
   cudaError err = call;												\
   if( cudaSuccess != err) {											\
       fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",	\
               __FILE__, __LINE__, cudaGetErrorString( err) );			\
   exit(EXIT_FAILURE);													\
   } } while (0)

#define THREADS_PER_BLOCK 512
#define MAXBLOCKS 65536
#define CUDATIME

float *work_mem_h;
float *coord_h;

float *work_mem_d;
float *coord_d;
int   *center_table_d;
bool  *switch_membership_d;
Point *p;

static int iter = 0;		// counter for total# of iteration


__device__ float
d_dist(int p1, int p2, int num, int dim, float *coord_d)
{
	float retval = 0.0;
	for(int i = 0; i < dim; i++){
		float tmp = coord_d[(i*num)+p1] - coord_d[(i*num)+p2];
		retval += tmp * tmp;
	}
	return retval;
}

__global__ void
kernel_compute_cost(int num, int dim, long x, Point *p, int K, int stride,
					float *coord_d, float *work_mem_d, int *center_table_d, bool *switch_membership_d)
{
	// block ID and global thread ID
	const int bid  = blockIdx.x + gridDim.x * blockIdx.y;
	const int tid = blockDim.x * bid + threadIdx.x;

	if(tid < num)
	{
		float *lower = &work_mem_d[tid*stride];
		
		// cost between this point and point[x]: euclidean distance multiplied by weight
		float x_cost = d_dist(tid, x, num, dim, coord_d) * p[tid].weight;
		
		// if computed cost is less then original (it saves), mark it as to reassign
		if ( x_cost < p[tid].cost )
		{
			switch_membership_d[tid] = 1;
			lower[K] += x_cost - p[tid].cost;
		}
		// if computed cost is larger, save the difference
		else
		{
			lower[center_table_d[p[tid].assign]] += p[tid].cost - x_cost;
		}
	}
}

void allocDevMem(int num, int dim)
{
	CUDA_SAFE_CALL( cudaMalloc((void**) &center_table_d,	  num * sizeof(int))   );
	CUDA_SAFE_CALL( cudaMalloc((void**) &switch_membership_d, num * sizeof(bool))  );
	CUDA_SAFE_CALL( cudaMalloc((void**) &p,					  num * sizeof(Point)) );
	CUDA_SAFE_CALL( cudaMalloc((void**) &coord_d,		num * dim * sizeof(float)) );
}

void allocHostMem(int num, int dim)
{
	coord_h	= (float*) malloc( num * dim * sizeof(float) );
}

void freeDevMem()
{
	CUDA_SAFE_CALL( cudaFree(center_table_d)	  );
	CUDA_SAFE_CALL( cudaFree(switch_membership_d) );
	CUDA_SAFE_CALL( cudaFree(p)					  );
	CUDA_SAFE_CALL( cudaFree(coord_d)			  );
}

void freeHostMem()
{
	free(coord_h);
}

float pgain( long x, Points *points, float z, long int *numcenters, int kmax, bool *is_center, int *center_table, bool *switch_membership, bool isCoordChanged,
							double *serial_t, double *cpu_to_gpu_t, double *gpu_to_cpu_t, double *alloc_t, double *kernel_t, double *free_t)
{	
#ifdef CUDATIME
	float tmp_t;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start, 0);
#endif

	cudaError_t error;
	
	int stride	= *numcenters + 1;			// size of each work_mem segment
	int K		= *numcenters ;				// number of centers
	int num		=  points->num;				// number of points
	int dim		=  points->dim;				// number of dimension
	int nThread =  num;						// number of threads == number of data points

	work_mem_h = (float*) malloc(stride * (nThread + 1) * sizeof(float) );
	if(iter == 0)
	{
		allocHostMem(num, dim);
	}
	
	int count = 0;
	for( int i=0; i<num; i++)
	{
		if( is_center[i] )
		{
			center_table[i] = count++;
		}
	}

	if(isCoordChanged || iter == 0)
	{
		for(int i=0; i<dim; i++)
		{
			for(int j=0; j<num; j++)
			{
				coord_h[ (num*i)+j ] = points->p[j].coord[i];
			}
		}
	}
	
#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*serial_t += (double) tmp_t;
	
	cudaEventRecord(start,0);
#endif

	CUDA_SAFE_CALL( cudaMalloc((void**) &work_mem_d,  stride * (nThread + 1) * sizeof(float)) );
	if( iter == 0 )
	{
		allocDevMem(num, dim);
	}
	
#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*alloc_t += (double) tmp_t;
	
	cudaEventRecord(start,0);
#endif

	if(isCoordChanged || iter == 0)
	{
		CUDA_SAFE_CALL( cudaMemcpy(coord_d,  coord_h,	 num * dim * sizeof(float), cudaMemcpyHostToDevice) );
	}
	CUDA_SAFE_CALL( cudaMemcpy(center_table_d,  center_table,  num * sizeof(int),   cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy(p,  points->p,				   num * sizeof(Point), cudaMemcpyHostToDevice) );
	
	CUDA_SAFE_CALL( cudaMemset((void*) switch_membership_d, 0,			num * sizeof(bool))  );
	CUDA_SAFE_CALL( cudaMemset((void*) work_mem_d,  		0, stride * (nThread + 1) * sizeof(float)) );
	
#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*cpu_to_gpu_t += (double) tmp_t;
	
	cudaEventRecord(start,0);
#endif
	
	// Determine the number of thread blocks in the x- and y-dimension
	int num_blocks 	 = (int) ((float) (num + THREADS_PER_BLOCK - 1) / (float) THREADS_PER_BLOCK);
	int num_blocks_y = (int) ((float) (num_blocks + MAXBLOCKS - 1)  / (float) MAXBLOCKS);
	int num_blocks_x = (int) ((float) (num_blocks+num_blocks_y - 1) / (float) num_blocks_y);	
	dim3 grid_size(num_blocks_x, num_blocks_y, 1);

	kernel_compute_cost<<<grid_size, THREADS_PER_BLOCK>>>(	
															num,					// in:	# of data
															dim,					// in:	dimension of point coordinates
															x,						// in:	point to open a center at
															p,						// in:	data point array
															K,						// in:	number of centers
															stride,					// in:  size of each work_mem segment
															coord_d,				// in:	array of point coordinates
															work_mem_d,				// out:	cost and lower field array
															center_table_d,			// in:	center index table
															switch_membership_d		// out:  changes in membership
															);
	cudaThreadSynchronize();
	
	// error check
	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		printf("kernel error: %s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
	
#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*kernel_t += (double) tmp_t;
	
	cudaEventRecord(start,0);
#endif
	
	CUDA_SAFE_CALL( cudaMemcpy(work_mem_h, 		  work_mem_d, 	stride * (nThread + 1) * sizeof(float), cudaMemcpyDeviceToHost) );
	CUDA_SAFE_CALL( cudaMemcpy(switch_membership, switch_membership_d,	 num * sizeof(bool),  cudaMemcpyDeviceToHost) );
	
#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*gpu_to_cpu_t += (double) tmp_t;
	
	cudaEventRecord(start,0);
#endif
	
	int number_of_centers_to_close = 0;
	float gl_cost_of_opening_x = z;
	float *gl_lower = &work_mem_h[stride * nThread];
	for(int i=0; i < num; i++)
	{
		if( is_center[i] )
		{
			float low = z;
		    for( int j = 0; j < num; j++ )
			{
				low += work_mem_h[ j*stride + center_table[i] ];
			}
			
		    gl_lower[center_table[i]] = low;
				
		    if ( low > 0 )
			{
				++number_of_centers_to_close;
				work_mem_h[i*stride+K] -= low;
		    }
		}
		gl_cost_of_opening_x += work_mem_h[i*stride+K];
	}

	if ( gl_cost_of_opening_x < 0 )
	{
		for(int i = 0; i < num; i++)
		{
			bool close_center = gl_lower[center_table[points->p[i].assign]] > 0 ;
			if ( switch_membership[i] || close_center )
			{
				points->p[i].cost = dist(points->p[i], points->p[x], dim) * points->p[i].weight;
				points->p[i].assign = x;
			}
		}
		
		for(int i = 0; i < num; i++)
		{
			if( is_center[i] && gl_lower[center_table[i]] > 0 )
			{
				is_center[i] = false;
			}
		}
		
		if( x >= 0 && x < num)
		{
			is_center[x] = true;
		}
		*numcenters = *numcenters + 1 - number_of_centers_to_close;
	}
	else
	{
		gl_cost_of_opening_x = 0;
	}

	free(work_mem_h);

	// free device memory
#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*serial_t += (double) tmp_t;
	
	cudaEventRecord(start,0);
#endif

	CUDA_SAFE_CALL( cudaFree(work_mem_d) );
	
#ifdef CUDATIME
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tmp_t, start, stop);
	*free_t += (double) tmp_t;
#endif
	iter++;
	return -gl_cost_of_opening_x;
}
