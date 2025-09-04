#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

#include <GL/glew.h>
#include <GL/glut.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "helper_cuda.h"
#include <cuda_gl_interop.h>
#include "bucketsort.cuh"
#include "bucketsort_kernel.cu"
#include "histogram1024_kernel.cu"

void calcPivotPoints(float *histogram, int histosize, int listsize, 
					 int divisions, float min, float max, float *pivotPoints, 
					 float histo_width);

const int histosize = 1024; 
unsigned int* h_offsets = NULL; 
unsigned int* d_offsets = NULL; 
int *d_indice = NULL; 
float *pivotPoints = NULL; 
float *historesult = NULL; 
float *l_pivotpoints = NULL; 
unsigned int *d_prefixoffsets = NULL; 
unsigned int *l_offsets = NULL; 

void init_bucketsort(int listsize)
{
	h_offsets = (unsigned int *) malloc(histosize * sizeof(int)); 
    checkCudaErrors(cudaMalloc((void**) &d_offsets, histosize * sizeof(unsigned int)));
	pivotPoints = (float *)malloc(DIVISIONS * sizeof(float)); 

    checkCudaErrors(cudaMalloc((void**) &d_indice, listsize * sizeof(int)));
	historesult = (float *)malloc(histosize * sizeof(float)); 

	checkCudaErrors(cudaMalloc((void**) &l_pivotpoints, DIVISIONS * sizeof(float))); 
	checkCudaErrors(cudaMalloc((void**) &l_offsets, DIVISIONS * sizeof(int))); 

	int blocks = ((listsize - 1) / (BUCKET_THREAD_N * BUCKET_BAND)) + 1; 
	checkCudaErrors(cudaMalloc((void**) &d_prefixoffsets, blocks * BUCKET_BLOCK_MEMORY * sizeof(int))); 

	initHistogram1024();
}

void finish_bucketsort()
{
    checkCudaErrors(cudaFree(d_indice));
	checkCudaErrors(cudaFree(d_offsets));
	checkCudaErrors(cudaFree(l_pivotpoints));
	checkCudaErrors(cudaFree(l_offsets)); 
	free(pivotPoints); 
	free(h_offsets);
	free(historesult);	
	checkCudaErrors(cudaFree(d_prefixoffsets)); 
	closeHistogram1024();
}

void bucketSort(float *d_input, float *d_output, int listsize,
				int *sizes, int *nullElements, float minimum, float maximum, 
				unsigned int *origOffsets)
{
	checkCudaErrors(cudaMemset((void *) d_offsets, 0, histosize * sizeof(int))); 
	histogram1024GPU(h_offsets, d_input, minimum, maximum, listsize); 
	for(int i=0; i<histosize; i++) historesult[i] = (float)h_offsets[i];

	calcPivotPoints(historesult, histosize, listsize, DIVISIONS, 
			minimum, maximum, pivotPoints,
			(maximum - minimum)/(float)histosize); 
	checkCudaErrors(cudaMemcpy(l_pivotpoints, pivotPoints, (DIVISIONS)*sizeof(int), cudaMemcpyHostToDevice)); 
	checkCudaErrors(cudaMemset((void *) d_offsets, 0, DIVISIONS * sizeof(int))); 
	checkCudaErrors(cudaBindTexture(0, texPivot, l_pivotpoints, DIVISIONS * sizeof(int))); 
    dim3 threads(BUCKET_THREAD_N, 1);
	int blocks = ((listsize - 1) / (threads.x * BUCKET_BAND)) + 1; 
    dim3 grid(blocks, 1);
	bucketcount <<< grid, threads >>>(d_input, d_indice, d_prefixoffsets, listsize);
#ifdef BUCKET_WG_SIZE_0
threads.x = BUCKET_WG_SIZE_0;
#else
	threads.x = 128;  
#endif
	grid.x = DIVISIONS / threads.x; 
	bucketprefixoffset <<< grid, threads >>>(d_prefixoffsets, d_offsets, blocks); 	

	cudaMemcpy(h_offsets, d_offsets, DIVISIONS * sizeof(int), cudaMemcpyDeviceToHost);

	origOffsets[0] = 0;
	for(int i=0; i<DIVISIONS; i++){
		origOffsets[i+1] = h_offsets[i] + origOffsets[i]; 
		if((h_offsets[i] % 4) != 0){
			nullElements[i] = (h_offsets[i] & ~3) + 4 - h_offsets[i]; 
		}
		else nullElements[i] = 0; 
	}
	for(int i=0; i<DIVISIONS; i++) sizes[i] = (h_offsets[i] + nullElements[i])/4; 
	for(int i=0; i<DIVISIONS; i++) {
		if((h_offsets[i] % 4) != 0)	h_offsets[i] = (h_offsets[i] & ~3) + 4; 
	}
	for(int i=1; i<DIVISIONS; i++) h_offsets[i] = h_offsets[i-1] + h_offsets[i]; 
	for(int i=DIVISIONS - 1; i>0; i--) h_offsets[i] = h_offsets[i-1]; 
	h_offsets[0] = 0; 
	cudaMemcpy(l_offsets, h_offsets, (DIVISIONS)*sizeof(int), cudaMemcpyHostToDevice); 
	cudaMemset(d_output, 0x0, (listsize + (DIVISIONS*4))*sizeof(float)); 
    threads.x = BUCKET_THREAD_N; 
	blocks = ((listsize - 1) / (threads.x * BUCKET_BAND)) + 1; 
    grid.x = blocks; 
	bucketsort <<< grid, threads >>>(d_input, d_indice, d_output, listsize, d_prefixoffsets, l_offsets);
}


void calcPivotPoints(float *histogram, int histosize, int listsize, 
					 int divisions, float min, float max, float *pivotPoints, float histo_width)
{
	float elemsPerSlice = listsize/(float)divisions; 
	float startsAt = min; 
	float endsAt = min + histo_width; 
	float we_need = elemsPerSlice; 
	int p_idx = 0; 
	for(int i=0; i<histosize; i++)
	{
		if(i == histosize - 1){
			if(!(p_idx < divisions)){
				pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
			}
			break; 
		}
		while(histogram[i] > we_need){
			if(!(p_idx < divisions)){
				printf("i=%d, p_idx = %d, divisions = %d\n", i, p_idx, divisions); 
				exit(0);
			}
			pivotPoints[p_idx++] = startsAt + (we_need/histogram[i]) * histo_width;
			startsAt += (we_need/histogram[i]) * histo_width; 
			histogram[i] -= we_need; 
			we_need = elemsPerSlice; 
		}
		we_need -= histogram[i]; 

		startsAt = endsAt; 
		endsAt += histo_width; 
	}
	while(p_idx < divisions){
		pivotPoints[p_idx] = pivotPoints[p_idx-1]; 
		p_idx++; 
	}
}

