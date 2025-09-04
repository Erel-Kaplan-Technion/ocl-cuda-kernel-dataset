#ifndef _BOXFILTER_KERNEL_CH_
#define _BOXFILTER_KERNEL_CH_

#include <helper_functions.h>
#include <helper_math.h>

cudaTextureObject_t tex;
cudaTextureObject_t texTempArray;
cudaTextureObject_t rgbaTex;
cudaTextureObject_t rgbaTexTempArray;
cudaArray          *d_array, *d_tempArray;

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

__device__ void d_boxfilter_x(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    t = id[0] * r;

    for (int x = 0; x < (r + 1); x++) {
        t += id[x];
    }

    od[0] = t * scale;

    for (int x = 1; x < (r + 1); x++) {
        t += id[x + r];
        t -= id[0];
        od[x] = t * scale;
    }

    for (int x = (r + 1); x < w - r; x++) {
        t += id[x + r];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }

    for (int x = w - r; x < w; x++) {
        t += id[w - 1];
        t -= id[x - r - 1];
        od[x] = t * scale;
    }
}

__device__ void d_boxfilter_y(float *id, float *od, int w, int h, int r)
{
    float scale = 1.0f / (float)((r << 1) + 1);

    float t;
    t = id[0] * r;

    for (int y = 0; y < (r + 1); y++) {
        t += id[y * w];
    }

    od[0] = t * scale;

    for (int y = 1; y < (r + 1); y++) {
        t += id[(y + r) * w];
        t -= id[0];
        od[y * w] = t * scale;
    }

    for (int y = (r + 1); y < (h - r); y++) {
        t += id[(y + r) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }

    for (int y = h - r; y < h; y++) {
        t += id[(h - 1) * w];
        t -= id[((y - r) * w) - w];
        od[y * w] = t * scale;
    }
}

__global__ void d_boxfilter_x_global(float *id, float *od, int w, int h, int r)
{
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;
    d_boxfilter_x(&id[y * w], &od[y * w], w, h, r);
}

__global__ void d_boxfilter_y_global(float *id, float *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    d_boxfilter_y(&id[x], &od[x], w, h, r);
}

__global__ void d_boxfilter_x_tex(float *od, int w, int h, int r, cudaTextureObject_t tex)
{
    float        scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y     = blockIdx.x * blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int x = -r; x <= r; x++) {
        t += tex2D<float>(tex, x, y);
    }

    od[y * w] = t * scale;

    for (int x = 1; x < w; x++) {
        t += tex2D<float>(tex, x + r, y);
        t -= tex2D<float>(tex, x - r - 1, y);
        od[y * w + x] = t * scale;
    }
}

__global__ void d_boxfilter_y_tex(float *od, int w, int h, int r, cudaTextureObject_t tex)
{
    float        scale = 1.0f / (float)((r << 1) + 1);
    unsigned int x     = blockIdx.x * blockDim.x + threadIdx.x;

    float t = 0.0f;

    for (int y = -r; y <= r; y++) {
        t += tex2D<float>(tex, x, y);
    }

    od[x] = t * scale;

    for (int y = 1; y < h; y++) {
        t += tex2D<float>(tex, x, y + r);
        t -= tex2D<float>(tex, x, y - r - 1);
        od[y * w + x] = t * scale;
    }
}

__device__ unsigned int rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return ((unsigned int)(rgba.w * 255.0f) << 24) | ((unsigned int)(rgba.z * 255.0f) << 16)
         | ((unsigned int)(rgba.y * 255.0f) << 8) | ((unsigned int)(rgba.x * 255.0f));
}

__device__ float4 rgbaIntToFloat(unsigned int c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;         //  /255.0f;
    rgba.y = ((c >> 8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c >> 16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c >> 24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

__global__ void d_boxfilter_rgba_x(unsigned int *od, int w, int h, int r, cudaTextureObject_t rgbaTex)
{
    float        scale = 1.0f / (float)((r << 1) + 1);
    unsigned int y     = blockIdx.x * blockDim.x + threadIdx.x;

    if (y < h) {
        float4 t = make_float4(0.0f);

        for (int x = -r; x <= r; x++) {
            t += tex2D<float4>(rgbaTex, x, y);
        }

        od[y * w] = rgbaFloatToInt(t * scale);

        for (int x = 1; x < w; x++) {
            t += tex2D<float4>(rgbaTex, x + r, y);
            t -= tex2D<float4>(rgbaTex, x - r - 1, y);
            od[y * w + x] = rgbaFloatToInt(t * scale);
        }
    }
}

__global__ void d_boxfilter_rgba_y(unsigned int *id, unsigned int *od, int w, int h, int r)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    id             = &id[x];
    od             = &od[x];

    float scale = 1.0f / (float)((r << 1) + 1);

    float4 t;
    t = rgbaIntToFloat(id[0]) * r;

    for (int y = 0; y < (r + 1); y++) {
        t += rgbaIntToFloat(id[y * w]);
    }

    od[0] = rgbaFloatToInt(t * scale);

    for (int y = 1; y < (r + 1); y++) {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[0]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    for (int y = (r + 1); y < (h - r); y++) {
        t += rgbaIntToFloat(id[(y + r) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }

    for (int y = h - r; y < h; y++) {
        t += rgbaIntToFloat(id[(h - 1) * w]);
        t -= rgbaIntToFloat(id[((y - r) * w) - w]);
        od[y * w] = rgbaFloatToInt(t * scale);
    }
}

extern "C" void initTexture(int width, int height, void *pImage, bool useRGBA)
{
    cudaChannelFormatDesc channelDesc;
    if (useRGBA) {
        channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
    }
    else {
        channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    }
    checkCudaErrors(cudaMallocArray(&d_array, &channelDesc, width, height));

    size_t bytesPerElem = (useRGBA ? sizeof(uchar4) : sizeof(float));
    checkCudaErrors(cudaMemcpy2DToArray(
        d_array, 0, 0, pImage, width * bytesPerElem, width * bytesPerElem, height, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMallocArray(&d_tempArray, &channelDesc, width, height));

    cudaResourceDesc texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = d_array;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0]   = cudaAddressModeWrap;
    texDescr.addressMode[1]   = cudaAddressModeWrap;
    texDescr.readMode         = cudaReadModeNormalizedFloat;

    checkCudaErrors(cudaCreateTextureObject(&rgbaTex, &texRes, &texDescr, NULL));

    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = d_tempArray;

    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0]   = cudaAddressModeClamp;
    texDescr.addressMode[1]   = cudaAddressModeClamp;
    texDescr.readMode         = cudaReadModeNormalizedFloat;

    checkCudaErrors(cudaCreateTextureObject(&rgbaTexTempArray, &texRes, &texDescr, NULL));

    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = d_array;

    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0]   = cudaAddressModeWrap;
    texDescr.addressMode[1]   = cudaAddressModeWrap;
    texDescr.readMode         = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&tex, &texRes, &texDescr, NULL));

    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = d_tempArray;

    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = true;
    texDescr.filterMode       = cudaFilterModePoint;
    texDescr.addressMode[0]   = cudaAddressModeWrap;
    texDescr.addressMode[1]   = cudaAddressModeWrap;
    texDescr.readMode         = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&texTempArray, &texRes, &texDescr, NULL));
}

extern "C" void freeTextures()
{
    checkCudaErrors(cudaDestroyTextureObject(tex));
    checkCudaErrors(cudaDestroyTextureObject(texTempArray));
    checkCudaErrors(cudaDestroyTextureObject(rgbaTex));
    checkCudaErrors(cudaDestroyTextureObject(rgbaTexTempArray));
    checkCudaErrors(cudaFreeArray(d_array));
    checkCudaErrors(cudaFreeArray(d_tempArray));
}
extern "C" double boxFilter(float              *d_temp,
                            float              *d_dest,
                            int                 width,
                            int                 height,
                            int                 radius,
                            int                 iterations,
                            int                 nthreads,
                            StopWatchInterface *timer)
{
    double dKernelTime = 0.0;

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < iterations; i++) {
        sdkResetTimer(&timer);
        if (iterations > 1) {
            d_boxfilter_x_tex<<<height / nthreads, nthreads, 0>>>(d_temp, width, height, radius, texTempArray);
        }
        else {
            d_boxfilter_x_tex<<<height / nthreads, nthreads, 0>>>(d_temp, width, height, radius, tex);
        }

        d_boxfilter_y_global<<<width / nthreads, nthreads, 0>>>(d_temp, d_dest, width, height, radius);

        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1) {
            checkCudaErrors(cudaMemcpy2DToArray(d_tempArray,
                                                0,
                                                0,
                                                d_dest,
                                                width * sizeof(float),
                                                width * sizeof(float),
                                                height,
                                                cudaMemcpyDeviceToDevice));
        }
    }

    return ((dKernelTime / 1000.) / (double)iterations);
}

extern "C" double boxFilterRGBA(unsigned int       *d_temp,
                                unsigned int       *d_dest,
                                int                 width,
                                int                 height,
                                int                 radius,
                                int                 iterations,
                                int                 nthreads,
                                StopWatchInterface *timer)
{
    double dKernelTime;

    for (int i = 0; i < iterations; i++) {
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        if (iterations > 1) {
            d_boxfilter_rgba_x<<<height / nthreads, nthreads, 0>>>(d_temp, width, height, radius, rgbaTexTempArray);
        }
        else {
            d_boxfilter_rgba_x<<<height / nthreads, nthreads, 0>>>(d_temp, width, height, radius, rgbaTex);
        }

        d_boxfilter_rgba_y<<<width / nthreads, nthreads, 0>>>(d_temp, d_dest, width, height, radius);

        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1) {
            checkCudaErrors(cudaMemcpy2DToArray(d_tempArray,
                                                0,
                                                0,
                                                d_dest,
                                                width * sizeof(unsigned int),
                                                width * sizeof(unsigned int),
                                                height,
                                                cudaMemcpyDeviceToDevice));
        }
    }

    return ((dKernelTime / 1000.) / (double)iterations);
}

#endif // #ifndef _BOXFILTER_KERNEL_H_
