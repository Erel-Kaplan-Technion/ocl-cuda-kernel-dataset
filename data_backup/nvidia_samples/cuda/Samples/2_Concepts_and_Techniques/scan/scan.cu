#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>

#include "scan_common.h"

#define THREADBLOCK_SIZE 256

inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    uint pos    = 2 * threadIdx.x - (threadIdx.x & (size - 1));
    s_Data[pos] = 0;
    pos += size;
    s_Data[pos] = idata;

    for (uint offset = 1; offset < size; offset <<= 1) {
        cg::sync(cta);
        uint t = s_Data[pos] + s_Data[pos - offset];
        cg::sync(cta);
        s_Data[pos] = t;
    }

    return s_Data[pos];
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    return scan1Inclusive(idata, s_Data, size, cta) - idata;
}

inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4, cta);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size, cg::thread_block cta)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size, cta);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

__global__ void scanExclusiveShared(uint4 *d_Dst, uint4 *d_Src, uint size)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint4 idata4 = d_Src[pos];

    uint4 odata4 = scan4Exclusive(idata4, s_Data, size, cta);

    d_Dst[pos] = odata4;
}

__global__ void scanExclusiveShared2(uint *d_Buf, uint *d_Dst, uint *d_Src, uint N, uint arrayLength)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    uint idata = 0;

    if (pos < N)
        idata = d_Dst[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos]
              + d_Src[(4 * THREADBLOCK_SIZE) - 1 + (4 * THREADBLOCK_SIZE) * pos];

    uint odata = scan1Exclusive(idata, s_Data, arrayLength, cta);

    if (pos < N) {
        d_Buf[pos] = odata;
    }
}

__global__ void uniformUpdate(uint4 *d_Data, uint *d_Buffer)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint  buf;
    uint             pos = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) {
        buf = d_Buffer[blockIdx.x];
    }

    cg::sync(cta);

    uint4 data4 = d_Data[pos];
    data4.x += buf;
    data4.y += buf;
    data4.z += buf;
    data4.w += buf;
    d_Data[pos] = data4;
}

extern "C" const uint MAX_BATCH_ELEMENTS   = 64 * 1048576;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;

static uint *d_Buf;

extern "C" void initScan(void)
{
    checkCudaErrors(cudaMalloc((void **)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(uint)));
}

extern "C" void closeScan(void) { checkCudaErrors(cudaFree(d_Buf)); }

static uint factorRadix2(uint &log2L, uint L)
{
    if (!L) {
        log2L = 0;
        return 0;
    }
    else {
        for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++)
            ;

        return L;
    }
}

static uint iDivUp(uint dividend, uint divisor)
{
    return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

extern "C" size_t scanExclusiveShort(uint *d_Dst, uint *d_Src, uint batchSize, uint arrayLength)
{
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);

    assert((arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE));

    assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

    assert((batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0);

    scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst, (uint4 *)d_Src, arrayLength);
    getLastCudaError("scanExclusiveShared() execution FAILED\n");

    return THREADBLOCK_SIZE;
}

extern "C" size_t scanExclusiveLarge(uint *d_Dst, uint *d_Src, uint batchSize, uint arrayLength)
{
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);

    assert((arrayLength >= MIN_LARGE_ARRAY_SIZE) && (arrayLength <= MAX_LARGE_ARRAY_SIZE));

    assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

    scanExclusiveShared<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>(
        (uint4 *)d_Dst, (uint4 *)d_Src, 4 * THREADBLOCK_SIZE);
    getLastCudaError("scanExclusiveShared() execution FAILED\n");

    const uint blockCount2 = iDivUp((batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE);
    scanExclusiveShared2<<<blockCount2, THREADBLOCK_SIZE>>>((uint *)d_Buf,
                                                            (uint *)d_Dst,
                                                            (uint *)d_Src,
                                                            (batchSize * arrayLength) / (4 * THREADBLOCK_SIZE),
                                                            arrayLength / (4 * THREADBLOCK_SIZE));
    getLastCudaError("scanExclusiveShared2() execution FAILED\n");

    uniformUpdate<<<(batchSize * arrayLength) / (4 * THREADBLOCK_SIZE), THREADBLOCK_SIZE>>>((uint4 *)d_Dst,
                                                                                            (uint *)d_Buf);
    getLastCudaError("uniformUpdate() execution FAILED\n");

    return THREADBLOCK_SIZE;
}
