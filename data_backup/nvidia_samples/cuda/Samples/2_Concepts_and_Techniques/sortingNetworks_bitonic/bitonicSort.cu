#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>

#include "sortingNetworks_common.cuh"
#include "sortingNetworks_common.h"

__global__ void
bitonicSortShared(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint arrayLength, uint dir)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0]                       = d_SrcKey[0];
    s_val[threadIdx.x + 0]                       = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < arrayLength; size <<= 1) {
        uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
        }
    }

    {
        for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], dir);
        }
    }

    cg::sync(cta);
    d_DstKey[0]                       = s_key[threadIdx.x + 0];
    d_DstVal[0]                       = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

__global__ void bitonicSortShared1(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0]                       = d_SrcKey[0];
    s_val[threadIdx.x + 0]                       = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
        uint ddd = (threadIdx.x & (size / 2)) != 0;

        for (uint stride = size / 2; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
        }
    }

    uint ddd = blockIdx.x & 1;
    {
        for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
            cg::sync(cta);
            uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
        }
    }

    cg::sync(cta);
    d_DstKey[0]                       = s_key[threadIdx.x + 0];
    d_DstVal[0]                       = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

__global__ void bitonicMergeGlobal(uint *d_DstKey,
                                   uint *d_DstVal,
                                   uint *d_SrcKey,
                                   uint *d_SrcVal,
                                   uint  arrayLength,
                                   uint  size,
                                   uint  stride,
                                   uint  dir)
{
    uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
    uint comparatorI        = global_comparatorI & (arrayLength / 2 - 1);

    uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
    uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

    uint keyA = d_SrcKey[pos + 0];
    uint valA = d_SrcVal[pos + 0];
    uint keyB = d_SrcKey[pos + stride];
    uint valB = d_SrcVal[pos + stride];

    Comparator(keyA, valA, keyB, valB, ddd);

    d_DstKey[pos + 0]      = keyA;
    d_DstVal[pos + 0]      = valA;
    d_DstKey[pos + stride] = keyB;
    d_DstVal[pos + stride] = valB;
}

__global__ void bitonicMergeShared(uint *d_DstKey,
                                   uint *d_DstVal,
                                   uint *d_SrcKey,
                                   uint *d_SrcVal,
                                   uint  arrayLength,
                                   uint  size,
                                   uint  dir)
{
    cg::thread_block cta = cg::this_thread_block();
    __shared__ uint s_key[SHARED_SIZE_LIMIT];
    __shared__ uint s_val[SHARED_SIZE_LIMIT];

    d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
    s_key[threadIdx.x + 0]                       = d_SrcKey[0];
    s_val[threadIdx.x + 0]                       = d_SrcVal[0];
    s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
    s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] = d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

    uint comparatorI = UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
    uint ddd         = dir ^ ((comparatorI & (size / 2)) != 0);

    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
        cg::sync(cta);
        uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
    }

    cg::sync(cta);
    d_DstKey[0]                       = s_key[threadIdx.x + 0];
    d_DstVal[0]                       = s_val[threadIdx.x + 0];
    d_DstKey[(SHARED_SIZE_LIMIT / 2)] = s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
    d_DstVal[(SHARED_SIZE_LIMIT / 2)] = s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

extern "C" uint factorRadix2(uint *log2L, uint L)
{
    if (!L) {
        *log2L = 0;
        return 0;
    }
    else {
        for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++)
            ;

        return L;
    }
}

extern "C" uint
bitonicSort(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey, uint *d_SrcVal, uint batchSize, uint arrayLength, uint dir)
{
    if (arrayLength < 2)
        return 0;

    uint log2L;
    uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
    assert(factorizationRemainder == 1);

    dir = (dir != 0);

    uint blockCount  = batchSize * arrayLength / SHARED_SIZE_LIMIT;
    uint threadCount = SHARED_SIZE_LIMIT / 2;

    if (arrayLength <= SHARED_SIZE_LIMIT) {
        assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
        bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal, arrayLength, dir);
    }
    else {
        bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey, d_SrcVal);

        for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
            for (unsigned stride = size / 2; stride > 0; stride >>= 1)
                if (stride >= SHARED_SIZE_LIMIT) {
                    bitonicMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(
                        d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride, dir);
                }
                else {
                    bitonicMergeShared<<<blockCount, threadCount>>>(
                        d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, dir);
                    break;
                }
    }

    return threadCount;
}
