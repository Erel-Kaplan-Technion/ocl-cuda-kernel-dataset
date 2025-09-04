#include <algorithm>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <iostream>

#include "FDTD3dGPU.h"
#include "FDTD3dGPUKernel.cuh"

bool getTargetDeviceGlobalMemSize(memsize_t *result, const int argc, const char **argv)
{
    int    deviceCount  = 0;
    int    targetDevice = 0;
    size_t memsize      = 0;

    printf(" cudaGetDeviceCount\n");
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    targetDevice = findCudaDevice(argc, (const char **)argv);

    printf(" cudaGetDeviceProperties\n");
    struct cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, targetDevice));

    memsize = deviceProp.totalGlobalMem;

    *result = (memsize_t)memsize;
    return true;
}

bool fdtdGPU(float       *output,
             const float *input,
             const float *coeff,
             const int    dimx,
             const int    dimy,
             const int    dimz,
             const int    radius,
             const int    timesteps,
             const int    argc,
             const char **argv)
{
    const int    outerDimx    = dimx + 2 * radius;
    const int    outerDimy    = dimy + 2 * radius;
    const int    outerDimz    = dimz + 2 * radius;
    const size_t volumeSize   = outerDimx * outerDimy * outerDimz;
    int          deviceCount  = 0;
    int          targetDevice = 0;
    float       *bufferOut    = 0;
    float       *bufferIn     = 0;
    dim3         dimBlock;
    dim3         dimGrid;

    const int    padding          = (128 / sizeof(float)) - radius;
    const size_t paddedVolumeSize = volumeSize + padding;

#ifdef GPU_PROFILING
    cudaEvent_t profileStart     = 0;
    cudaEvent_t profileEnd       = 0;
    const int   profileTimesteps = timesteps - 1;

    if (profileTimesteps < 1) {
        printf(" cannot profile with fewer than two timesteps (timesteps=%d), "
               "profiling is disabled.\n",
               timesteps);
    }

#endif

    if (radius != RADIUS) {
        printf("radius is invalid, must be %d - see kernel for details.\n", RADIUS);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    targetDevice = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaSetDevice(targetDevice));

    checkCudaErrors(cudaMalloc((void **)&bufferOut, paddedVolumeSize * sizeof(float)));
    checkCudaErrors(cudaMalloc((void **)&bufferIn, paddedVolumeSize * sizeof(float)));

    int userBlockSize;

    if (checkCmdLineFlag(argc, (const char **)argv, "block-size")) {
        userBlockSize = getCmdLineArgumentInt(argc, argv, "block-size");
        userBlockSize = (userBlockSize / k_blockDimX * k_blockDimX);

        userBlockSize = MIN(MAX(userBlockSize, k_blockSizeMin), k_blockSizeMax);
    }
    else {
        userBlockSize = k_blockSizeMax;
    }

    struct cudaFuncAttributes funcAttrib;
    checkCudaErrors(cudaFuncGetAttributes(&funcAttrib, FiniteDifferencesKernel));

    userBlockSize = MIN(userBlockSize, funcAttrib.maxThreadsPerBlock);

    dimBlock.x = k_blockDimX;
    dimBlock.y = ((userBlockSize / k_blockDimX) < (size_t)k_blockDimMaxY) ? (userBlockSize / k_blockDimX)
                                                                          : (size_t)k_blockDimMaxY;
    dimGrid.x  = (unsigned int)ceil((float)dimx / dimBlock.x);
    dimGrid.y  = (unsigned int)ceil((float)dimy / dimBlock.y);
    printf(" set block size to %dx%d\n", dimBlock.x, dimBlock.y);
    printf(" set grid size to %dx%d\n", dimGrid.x, dimGrid.y);

    if (dimBlock.x < RADIUS || dimBlock.y < RADIUS) {
        printf("invalid block size, x (%d) and y (%d) must be >= radius (%d).\n", dimBlock.x, dimBlock.y, RADIUS);
        exit(EXIT_FAILURE);
    }

    checkCudaErrors(cudaMemcpy(bufferIn + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(bufferOut + padding, input, volumeSize * sizeof(float), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpyToSymbol(stencil, (void *)coeff, (radius + 1) * sizeof(float)));

#ifdef GPU_PROFILING

    checkCudaErrors(cudaEventCreate(&profileStart));
    checkCudaErrors(cudaEventCreate(&profileEnd));

#endif

    float *bufferSrc = bufferIn + padding;
    float *bufferDst = bufferOut + padding;
    printf(" GPU FDTD loop\n");

#ifdef GPU_PROFILING
    checkCudaErrors(cudaEventRecord(profileStart, 0));
#endif

    for (int it = 0; it < timesteps; it++) {
        printf("\tt = %d ", it);

        printf("launch kernel\n");
        FiniteDifferencesKernel<<<dimGrid, dimBlock>>>(bufferDst, bufferSrc, dimx, dimy, dimz);

        float *tmp = bufferDst;
        bufferDst  = bufferSrc;
        bufferSrc  = tmp;
    }

    printf("\n");

#ifdef GPU_PROFILING
    checkCudaErrors(cudaEventRecord(profileEnd, 0));
#endif

    checkCudaErrors(cudaDeviceSynchronize());

    checkCudaErrors(cudaMemcpy(output, bufferSrc, volumeSize * sizeof(float), cudaMemcpyDeviceToHost));

#ifdef GPU_PROFILING
    float elapsedTimeMS = 0;

    if (profileTimesteps > 0) {
        checkCudaErrors(cudaEventElapsedTime(&elapsedTimeMS, profileStart, profileEnd));
    }

    if (profileTimesteps > 0) {
        double elapsedTime    = elapsedTimeMS * 1.0e-3;
        double avgElapsedTime = elapsedTime / (double)profileTimesteps;
        size_t pointsComputed = dimx * dimy * dimz;
        double throughputM = 1.0e-6 * (double)pointsComputed / avgElapsedTime;
        printf("FDTD3d, Throughput = %.4f MPoints/s, Time = %.5f s, Size = %u Points, "
               "NumDevsUsed = %u, Blocksize = %u\n",
               throughputM,
               avgElapsedTime,
               pointsComputed,
               1,
               dimBlock.x * dimBlock.y);
    }

#endif

    if (bufferIn) {
        checkCudaErrors(cudaFree(bufferIn));
    }

    if (bufferOut) {
        checkCudaErrors(cudaFree(bufferOut));
    }

#ifdef GPU_PROFILING

    if (profileStart) {
        checkCudaErrors(cudaEventDestroy(profileStart));
    }

    if (profileEnd) {
        checkCudaErrors(cudaEventDestroy(profileEnd));
    }

#endif
    return true;
}
