#ifndef _KMEANS_CUDA_KERNEL_H_
#define _KMEANS_CUDA_KERNEL_H_

#include <stdio.h>
#include <cuda.h>

#include "kmeans.h"

#define ASSUMED_NR_CLUSTERS 32

#define SDATA( index)      CUT_BANK_CHECKER(sdata, index)

texture<float, 1, cudaReadModeElementType> t_features;
texture<float, 1, cudaReadModeElementType> t_features_flipped;
texture<float, 1, cudaReadModeElementType> t_clusters;


__constant__ float c_clusters[ASSUMED_NR_CLUSTERS*34];


__global__ void invert_mapping(float *input,
							   float *output,
							   int npoints,
							   int nfeatures)
{
	int point_id = threadIdx.x + blockDim.x*blockIdx.x;
	int i;

	if(point_id < npoints){
		for(i=0;i<nfeatures;i++)
			output[point_id + npoints*i] = input[point_id*nfeatures + i];
	}
	return;
}


__global__ void
kmeansPoint(float  *features,
            int     nfeatures,
            int     npoints,
            int     nclusters,
            int    *membership,
			float  *clusters,
			float  *block_clusters,
			int    *block_deltas) 
{

	const unsigned int block_id = gridDim.x*blockIdx.y+blockIdx.x;
	const unsigned int point_id = block_id*blockDim.x*blockDim.y + threadIdx.x;
  
	int  index = -1;

	if (point_id < npoints)
	{
		int i, j;
		float min_dist = FLT_MAX;
		float dist;
		
		for (i=0; i<nclusters; i++) {
			int cluster_base_index = i*nfeatures;
			float ans=0.0;

			for (j=0; j < nfeatures; j++)
			{					
				int addr = point_id + j*npoints;
				float diff = (tex1Dfetch(t_features,addr) -
							  c_clusters[cluster_base_index + j]);
				ans += diff*diff;
			}
			dist = ans;		

			if (dist < min_dist) {
				min_dist = dist;
				index    = i;
			}
		}
	}
	

#ifdef GPU_DELTA_REDUCTION
	__shared__ int deltas[THREADS_PER_BLOCK];
	if(threadIdx.x < THREADS_PER_BLOCK) {
		deltas[threadIdx.x] = 0;
	}
#endif
	if (point_id < npoints)
	{
#ifdef GPU_DELTA_REDUCTION
		if (membership[point_id] != index) {
			deltas[threadIdx.x] = 1;
		}
#endif
		membership[point_id] = index;
	}

#ifdef GPU_DELTA_REDUCTION
	__syncthreads();

	unsigned int threadids_participating = THREADS_PER_BLOCK / 2;
	for(;threadids_participating > 1; threadids_participating /= 2) {
   		if(threadIdx.x < threadids_participating) {
			deltas[threadIdx.x] += deltas[threadIdx.x + threadids_participating];
		}
   		__syncthreads();
	}
	if(threadIdx.x < 1)	{deltas[threadIdx.x] += deltas[threadIdx.x + 1];}
	__syncthreads();
	if(threadIdx.x == 0) {
		block_deltas[blockIdx.y * gridDim.x + blockIdx.x] = deltas[0];
		
	}

#endif


#ifdef GPU_NEW_CENTER_REDUCTION
	int center_id = threadIdx.x / nfeatures;    
	int dim_id = threadIdx.x - nfeatures*center_id;

	__shared__ int new_center_ids[THREADS_PER_BLOCK];

	new_center_ids[threadIdx.x] = index;
	__syncthreads();

	int new_base_index = (point_id - threadIdx.x)*nfeatures + dim_id;
	float accumulator = 0.f;

	if(threadIdx.x < nfeatures * nclusters) {
		for(int i = 0; i< (THREADS_PER_BLOCK); i++) {
			float val = tex1Dfetch(t_features_flipped,new_base_index+i*nfeatures);
			if(new_center_ids[i] == center_id) 
				accumulator += val;
		}
	
		block_clusters[(blockIdx.y*gridDim.x + blockIdx.x) * nclusters * nfeatures + threadIdx.x] = accumulator;
	}
#endif

}
#endif // #ifndef _KMEANS_CUDA_KERNEL_H_
