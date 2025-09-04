#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include <helper_image.h>
#include <helper_string.h>

const char *sSDKsample = "Transpose";

#define TILE_DIM   32
#define BLOCK_ROWS 16

int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;
int MUL_FACTOR    = TILE_DIM;

#define FLOOR(a, b) (a - (a % b))

int MAX_TILES = (FLOOR(MATRIX_SIZE_X, 512) * FLOOR(MATRIX_SIZE_Y, 512)) / (TILE_DIM * TILE_DIM);

#define NUM_REPS 100

__global__ void copy(float *odata, float *idata, int width, int height)
{
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index = xIndex + width * yIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index + i * width] = idata[index + i * width];
    }
}

__global__ void copySharedMem(float *odata, float *idata, int width, int height)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index = xIndex + width * yIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < width && yIndex < height) {
            tile[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
        }
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        if (xIndex < height && yIndex < width) {
            odata[index + i * width] = tile[threadIdx.y + i][threadIdx.x];
        }
    }
}

__global__ void transposeNaive(float *odata, float *idata, int width, int height)
{
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

    int index_in  = xIndex + width * yIndex;
    int index_out = yIndex + height * xIndex;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i] = idata[index_in + i * width];
    }
}

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int xIndex   = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int xIndex   = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void transposeDiagonal(float *odata, float *idata, int width, int height)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int blockIdx_x, blockIdx_y;

    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    }
    else {
        int bid    = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    int xIndex   = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void transposeFineGrained(float *odata, float *idata, int width, int height)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float block[TILE_DIM][TILE_DIM + 1];

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index  = xIndex + (yIndex)*width;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        block[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index + i * height] = block[threadIdx.x][threadIdx.y + i];
    }
}

__global__ void transposeCoarseGrained(float *odata, float *idata, int width, int height)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ float block[TILE_DIM][TILE_DIM + 1];

    int xIndex   = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex   = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex        = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex        = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        block[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }

    cg::sync(cta);

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
        odata[index_out + i * height] = block[threadIdx.y + i][threadIdx.x];
    }
}

void computeTransposeGold(float *gold, float *idata, const int size_x, const int size_y)
{
    for (int y = 0; y < size_y; ++y) {
        for (int x = 0; x < size_x; ++x) {
            gold[(x * size_y) + y] = idata[(y * size_x) + x];
        }
    }
}

void getParams(int argc, char **argv, cudaDeviceProp &deviceProp, int &size_x, int &size_y, int max_tile_dim)
{
    if (checkCmdLineFlag(argc, (const char **)argv, "dimX")) {
        size_x = getCmdLineArgumentInt(argc, (const char **)argv, "dimX");

        if (size_x > max_tile_dim) {
            printf("> MatrixSize X = %d is greater than the recommended size = %d\n", size_x, max_tile_dim);
        }
        else {
            printf("> MatrixSize X = %d\n", size_x);
        }
    }
    else {
        size_x = max_tile_dim;
        size_x = FLOOR(size_x, 512);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "dimY")) {
        size_y = getCmdLineArgumentInt(argc, (const char **)argv, "dimY");

        if (size_y > max_tile_dim) {
            printf("> MatrixSize Y = %d is greater than the recommended size = %d\n", size_y, max_tile_dim);
        }
        else {
            printf("> MatrixSize Y = %d\n", size_y);
        }
    }
    else {
        size_y = max_tile_dim;
        size_y = FLOOR(size_y, 512);
    }
}

void showHelp()
{
    printf("\n%s : Command line options\n", sSDKsample);
    printf("\t-device=n          (where n=0,1,2.... for the GPU device)\n\n");
    printf("> The default matrix size can be overridden with these parameters\n");
    printf("\t-dimX=row_dim_size (matrix row    dimensions)\n");
    printf("\t-dimY=col_dim_size (matrix column dimensions)\n");
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", sSDKsample);

    if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
        showHelp();
        return 0;
    }

    int            devID = findCudaDevice(argc, (const char **)argv);
    cudaDeviceProp deviceProp;

    checkCudaErrors(cudaGetDevice(&devID));
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

    float scale_factor, total_tiles;
    scale_factor = max(
        (192.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)),
        1.0f);

    printf("> Device %d: \"%s\"\n", devID, deviceProp.name);
    printf("> SM Capability %d.%d detected:\n", deviceProp.major, deviceProp.minor);

    int size_x, size_y, max_matrix_dim, matrix_size_test;

    matrix_size_test = 512;
    total_tiles      = (float)MAX_TILES / scale_factor;

    max_matrix_dim = FLOOR((int)(floor(sqrt(total_tiles)) * TILE_DIM), matrix_size_test);

    if (max_matrix_dim == 0) {
        max_matrix_dim = matrix_size_test;
    }

    printf("> [%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n",
           deviceProp.name,
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    printf("> Compute performance scaling factor = %4.2f\n", scale_factor);

    getParams(argc, argv, deviceProp, size_x, size_y, max_matrix_dim);

    if (size_x != size_y) {
        printf("\n[%s] does not support non-square matrices (row_dim_size(%d) != "
               "col_dim_size(%d))\nExiting...\n\n",
               sSDKsample,
               size_x,
               size_y);
        exit(EXIT_FAILURE);
    }

    if (size_x % TILE_DIM != 0 || size_y % TILE_DIM != 0) {
        printf("[%s] Matrix size must be integral multiple of tile "
               "size\nExiting...\n\n",
               sSDKsample);
        exit(EXIT_FAILURE);
    }

    void (*kernel)(float *, float *, int, int);
    const char *kernelName;

    dim3 grid(size_x / TILE_DIM, size_y / TILE_DIM), threads(TILE_DIM, BLOCK_ROWS);

    if (grid.x < 1 || grid.y < 1) {
        printf("[%s] grid size computation incorrect in test \nExiting...\n\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    cudaEvent_t start, stop;

    size_t mem_size = static_cast<size_t>(sizeof(float) * size_x * size_y);

    if (2 * mem_size > deviceProp.totalGlobalMem) {
        printf("Input matrix size is larger than the available device memory!\n");
        printf("Please choose a smaller size matrix\n");
        exit(EXIT_FAILURE);
    }

    float *h_idata       = (float *)malloc(mem_size);
    float *h_odata       = (float *)malloc(mem_size);
    float *transposeGold = (float *)malloc(mem_size);
    float *gold;

    float *d_idata, *d_odata;
    checkCudaErrors(cudaMalloc((void **)&d_idata, mem_size));
    checkCudaErrors(cudaMalloc((void **)&d_odata, mem_size));

    for (int i = 0; i < (size_x * size_y); ++i) {
        h_idata[i] = (float)i;
    }

    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    computeTransposeGold(transposeGold, h_idata, size_x, size_y);

    printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: "
           "%dx%d\n\n",
           size_x,
           size_y,
           size_x / TILE_DIM,
           size_y / TILE_DIM,
           TILE_DIM,
           TILE_DIM,
           TILE_DIM,
           BLOCK_ROWS);

    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    bool success = true;

    for (int k = 0; k < 8; k++) {
        switch (k) {
        case 0:
            kernel     = &copy;
            kernelName = "simple copy       ";
            break;

        case 1:
            kernel     = &copySharedMem;
            kernelName = "shared memory copy";
            break;

        case 2:
            kernel     = &transposeNaive;
            kernelName = "naive             ";
            break;

        case 3:
            kernel     = &transposeCoalesced;
            kernelName = "coalesced         ";
            break;

        case 4:
            kernel     = &transposeNoBankConflicts;
            kernelName = "optimized         ";
            break;

        case 5:
            kernel     = &transposeCoarseGrained;
            kernelName = "coarse-grained    ";
            break;

        case 6:
            kernel     = &transposeFineGrained;
            kernelName = "fine-grained      ";
            break;

        case 7:
            kernel     = &transposeDiagonal;
            kernelName = "diagonal          ";
            break;
        }

        if (kernel == &copy || kernel == &copySharedMem) {
            gold = h_idata;
        }
        else if (kernel == &transposeCoarseGrained || kernel == &transposeFineGrained) {
            gold = h_odata;
        }
        else {
            gold = transposeGold;
        }

        checkCudaErrors(cudaGetLastError());

        kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y);

        checkCudaErrors(cudaEventRecord(start, 0));

        for (int i = 0; i < NUM_REPS; i++) {
            kernel<<<grid, threads>>>(d_odata, d_idata, size_x, size_y);
            checkCudaErrors(cudaGetLastError());
        }

        checkCudaErrors(cudaEventRecord(stop, 0));
        checkCudaErrors(cudaEventSynchronize(stop));
        float kernelTime;
        checkCudaErrors(cudaEventElapsedTime(&kernelTime, start, stop));

        checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
        bool res = compareData(gold, h_odata, size_x * size_y, 0.01f, 0.0f);

        if (res == false) {
            printf("*** %s kernel FAILED ***\n", kernelName);
            success = false;
        }

        checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));
        res = compareData(gold, h_odata, size_x * size_y, 0.01f, 0.0f);

        if (res == false) {
            printf("*** %s kernel FAILED ***\n", kernelName);
            success = false;
        }

        float kernelBandwidth = 2.0f * 1000.0f * mem_size / (1024 * 1024 * 1024) / (kernelTime / NUM_REPS);
        printf("transpose %s, Throughput = %.4f GB/s, Time = %.5f ms, Size = %u fp32 "
               "elements, NumDevsUsed = %u, Workgroup = %u\n",
               kernelName,
               kernelBandwidth,
               kernelTime / NUM_REPS,
               (size_x * size_y),
               1,
               TILE_DIM * BLOCK_ROWS);

        for (int i = 0; i < (size_x * size_y); ++i) {
            h_odata[i] = 0;
        }

        checkCudaErrors(cudaMemcpy(d_odata, h_odata, mem_size, cudaMemcpyHostToDevice));
    }

    free(h_idata);
    free(h_odata);
    free(transposeGold);
    cudaFree(d_idata);
    cudaFree(d_odata);

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));

    if (!success) {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
