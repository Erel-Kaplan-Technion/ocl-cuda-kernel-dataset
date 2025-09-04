#include <assert.h>
#include <stdio.h>

#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

#ifndef MAX
#define MAX(a, b) (a > b ? a : b)
#endif

#include "simpleMultiGPU.h"

const int MAX_GPU_COUNT = 32;
const int DATA_N        = 1048576 * 32;

__global__ static void reduceKernel(float *d_Result, float *d_Input, int N)
{
    const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
    const int threadN = gridDim.x * blockDim.x;
    float     sum     = 0;

    for (int pos = tid; pos < N; pos += threadN)
        sum += d_Input[pos];

    d_Result[tid] = sum;
}

int main(int argc, char **argv)
{
    TGPUplan plan[MAX_GPU_COUNT];

    float h_SumGPU[MAX_GPU_COUNT];

    float  sumGPU;
    double sumCPU, diff;

    int i, j, gpuBase, GPU_N;

    const int BLOCK_N  = 32;
    const int THREAD_N = 256;
    const int ACCUM_N  = BLOCK_N * THREAD_N;

    printf("Starting simpleMultiGPU\n");
    checkCudaErrors(cudaGetDeviceCount(&GPU_N));

    if (GPU_N > MAX_GPU_COUNT) {
        GPU_N = MAX_GPU_COUNT;
    }

    printf("CUDA-capable device count: %i\n", GPU_N);

    printf("Generating input data...\n\n");

    for (i = 0; i < GPU_N; i++) {
        plan[i].dataN = DATA_N / GPU_N;
    }

    for (i = 0; i < DATA_N % GPU_N; i++) {
        plan[i].dataN++;
    }

    gpuBase = 0;

    for (i = 0; i < GPU_N; i++) {
        plan[i].h_Sum = h_SumGPU + i;
        gpuBase += plan[i].dataN;
    }

    for (i = 0; i < GPU_N; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaStreamCreate(&plan[i].stream));
        checkCudaErrors(cudaMalloc((void **)&plan[i].d_Data, plan[i].dataN * sizeof(float)));
        checkCudaErrors(cudaMalloc((void **)&plan[i].d_Sum, ACCUM_N * sizeof(float)));
        checkCudaErrors(cudaMallocHost((void **)&plan[i].h_Sum_from_device, ACCUM_N * sizeof(float)));
        checkCudaErrors(cudaMallocHost((void **)&plan[i].h_Data, plan[i].dataN * sizeof(float)));

        for (j = 0; j < plan[i].dataN; j++) {
            plan[i].h_Data[j] = (float)rand() / (float)RAND_MAX;
        }
    }

    printf("Computing with %d GPUs...\n", GPU_N);
    StopWatchInterface *timer = NULL;
    sdkCreateTimer(&timer);

    sdkStartTimer(&timer);

    for (i = 0; i < GPU_N; i++) {
        checkCudaErrors(cudaSetDevice(i));

        checkCudaErrors(cudaMemcpyAsync(
            plan[i].d_Data, plan[i].h_Data, plan[i].dataN * sizeof(float), cudaMemcpyHostToDevice, plan[i].stream));

        reduceKernel<<<BLOCK_N, THREAD_N, 0, plan[i].stream>>>(plan[i].d_Sum, plan[i].d_Data, plan[i].dataN);
        getLastCudaError("reduceKernel() execution failed.\n");

        checkCudaErrors(cudaMemcpyAsync(
            plan[i].h_Sum_from_device, plan[i].d_Sum, ACCUM_N * sizeof(float), cudaMemcpyDeviceToHost, plan[i].stream));
    }

    for (i = 0; i < GPU_N; i++) {
        float sum;

        checkCudaErrors(cudaSetDevice(i));

        cudaStreamSynchronize(plan[i].stream);

        sum = 0;

        for (j = 0; j < ACCUM_N; j++) {
            sum += plan[i].h_Sum_from_device[j];
        }

        *(plan[i].h_Sum) = (float)sum;

        checkCudaErrors(cudaFreeHost(plan[i].h_Sum_from_device));
        checkCudaErrors(cudaFree(plan[i].d_Sum));
        checkCudaErrors(cudaFree(plan[i].d_Data));
        checkCudaErrors(cudaStreamDestroy(plan[i].stream));
    }

    sumGPU = 0;

    for (i = 0; i < GPU_N; i++) {
        sumGPU += h_SumGPU[i];
    }

    sdkStopTimer(&timer);
    printf("  GPU Processing time: %f (ms)\n\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    printf("Computing with Host CPU...\n\n");

    sumCPU = 0;

    for (i = 0; i < GPU_N; i++) {
        for (j = 0; j < plan[i].dataN; j++) {
            sumCPU += plan[i].h_Data[j];
        }
    }

    printf("Comparing GPU and Host CPU results...\n");
    diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
    printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
    printf("  Relative difference: %E \n\n", diff);

    for (i = 0; i < GPU_N; i++) {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaFreeHost(plan[i].h_Data));
    }

    exit((diff < 1e-5) ? EXIT_SUCCESS : EXIT_FAILURE);
}
