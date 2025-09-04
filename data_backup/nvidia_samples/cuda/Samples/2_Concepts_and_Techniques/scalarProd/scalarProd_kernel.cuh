#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#define IMUL(a, b) __mul24(a, b)


#define ACCUM_N 1024
__global__ void scalarProdGPU(float *d_C, float *d_A, float *d_B, int vectorN, int elementN)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    // Accumulators cache
    __shared__ float accumResult[ACCUM_N];
    for (int vec = blockIdx.x; vec < vectorN; vec += gridDim.x) {
        int vectorBase = IMUL(elementN, vec);
        int vectorEnd  = vectorBase + elementN;

        for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x) {
            float sum = 0;

            for (int pos = vectorBase + iAccum; pos < vectorEnd; pos += ACCUM_N)
                sum += d_A[pos] * d_B[pos];

            accumResult[iAccum] = sum;
        }

        for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1) {
            cg::sync(cta);

            for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x)
                accumResult[iAccum] += accumResult[stride + iAccum];
        }

        cg::sync(cta);

        if (threadIdx.x == 0)
            d_C[vec] = accumResult[0];
    }
}
