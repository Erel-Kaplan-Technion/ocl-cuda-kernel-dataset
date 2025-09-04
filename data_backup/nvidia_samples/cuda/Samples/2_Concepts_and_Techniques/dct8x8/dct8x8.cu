#include "BmpUtil.h"
#include "Common.h"
#include "DCT8x8_Gold.h"

#define BENCHMARK_SIZE 10

#define PSNR_THRESHOLD_EQUAL 40

#include "dct8x8_kernel1.cuh"
#include "dct8x8_kernel2.cuh"
#include "dct8x8_kernel_quantization.cuh"
#include "dct8x8_kernel_short.cuh"


float WrapperGold1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
    int    StrideF;
    float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);
    float *ImgF2 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

    CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
    AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

    StopWatchInterface *timerGold = 0;
    sdkCreateTimer(&timerGold);
    sdkResetTimer(&timerGold);

    for (int i = 0; i < BENCHMARK_SIZE; i++) {
        sdkStartTimer(&timerGold);
        computeDCT8x8Gold1(ImgF1, ImgF2, StrideF, Size);
        sdkStopTimer(&timerGold);
    }

    float TimerGoldSpan = sdkGetAverageTimerValue(&timerGold);
    sdkDeleteTimer(&timerGold);

    quantizeGoldFloat(ImgF2, StrideF, Size);

    computeIDCT8x8Gold1(ImgF2, ImgF1, StrideF, Size);

    AddFloatPlane(128.0f, ImgF1, StrideF, Size);
    CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

    FreePlane(ImgF1);
    FreePlane(ImgF2);

    return TimerGoldSpan;
}


float WrapperGold2(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
    int    StrideF;
    float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);
    float *ImgF2 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

    CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
    AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

    StopWatchInterface *timerGold = 0;
    sdkCreateTimer(&timerGold);
    sdkResetTimer(&timerGold);

    for (int i = 0; i < BENCHMARK_SIZE; i++) {
        sdkStartTimer(&timerGold);
        computeDCT8x8Gold2(ImgF1, ImgF2, StrideF, Size);
        sdkStopTimer(&timerGold);
    }

    float TimerGoldSpan = sdkGetAverageTimerValue(&timerGold);
    sdkDeleteTimer(&timerGold);

    quantizeGoldFloat(ImgF2, StrideF, Size);

    computeIDCT8x8Gold2(ImgF2, ImgF1, StrideF, Size);

    AddFloatPlane(128.0f, ImgF1, StrideF, Size);
    CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

    FreePlane(ImgF1);
    FreePlane(ImgF2);

    return TimerGoldSpan;
}


float WrapperCUDA1(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
    cudaChannelFormatDesc floattex = cudaCreateChannelDesc<float>();

    cudaArray *Src;
    float     *Dst;
    size_t     DstStride;
    checkCudaErrors(cudaMallocArray(&Src, &floattex, Size.width, Size.height));
    checkCudaErrors(cudaMallocPitch((void **)(&Dst), &DstStride, Size.width * sizeof(float), Size.height));
    DstStride /= sizeof(float);

    int    ImgSrcFStride;
    float *ImgSrcF = MallocPlaneFloat(Size.width, Size.height, &ImgSrcFStride);
    CopyByte2Float(ImgSrc, Stride, ImgSrcF, ImgSrcFStride, Size);
    AddFloatPlane(-128.0f, ImgSrcF, ImgSrcFStride, Size);

    checkCudaErrors(cudaMemcpy2DToArray(Src,
                                        0,
                                        0,
                                        ImgSrcF,
                                        ImgSrcFStride * sizeof(float),
                                        Size.width * sizeof(float),
                                        Size.height,
                                        cudaMemcpyHostToDevice));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    StopWatchInterface *timerCUDA = 0;
    sdkCreateTimer(&timerCUDA);
    sdkResetTimer(&timerCUDA);

    cudaTextureObject_t TexSrc;
    cudaResourceDesc    texRes;
    memset(&texRes, 0, sizeof(cudaResourceDesc));

    texRes.resType         = cudaResourceTypeArray;
    texRes.res.array.array = Src;

    cudaTextureDesc texDescr;
    memset(&texDescr, 0, sizeof(cudaTextureDesc));

    texDescr.normalizedCoords = false;
    texDescr.filterMode       = cudaFilterModeLinear;
    texDescr.addressMode[0]   = cudaAddressModeWrap;
    texDescr.addressMode[1]   = cudaAddressModeWrap;
    texDescr.readMode         = cudaReadModeElementType;

    checkCudaErrors(cudaCreateTextureObject(&TexSrc, &texRes, &texDescr, NULL));

    for (int i = 0; i < BENCHMARK_SIZE; i++) {
        sdkStartTimer(&timerCUDA);
        CUDAkernel1DCT<<<grid, threads>>>(Dst, (int)DstStride, 0, 0, TexSrc);
        checkCudaErrors(cudaDeviceSynchronize());
        sdkStopTimer(&timerCUDA);
    }

    getLastCudaError("Kernel execution failed");

    float TimerCUDASpan = sdkGetAverageTimerValue(&timerCUDA);
    sdkDeleteTimer(&timerCUDA);

    CUDAkernelQuantizationFloat<<<grid, threads>>>(Dst, (int)DstStride);
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpy2DToArray(
        Src, 0, 0, Dst, DstStride * sizeof(float), Size.width * sizeof(float), Size.height, cudaMemcpyDeviceToDevice));

    CUDAkernel1IDCT<<<grid, threads>>>(Dst, (int)DstStride, 0, 0, TexSrc);
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpy2D(ImgSrcF,
                                 ImgSrcFStride * sizeof(float),
                                 Dst,
                                 DstStride * sizeof(float),
                                 Size.width * sizeof(float),
                                 Size.height,
                                 cudaMemcpyDeviceToHost));

    AddFloatPlane(128.0f, ImgSrcF, ImgSrcFStride, Size);
    CopyFloat2Byte(ImgSrcF, ImgSrcFStride, ImgDst, Stride, Size);

    checkCudaErrors(cudaDestroyTextureObject(TexSrc));
    checkCudaErrors(cudaFreeArray(Src));
    checkCudaErrors(cudaFree(Dst));
    FreePlane(ImgSrcF);

    return TimerCUDASpan;
}



float WrapperCUDA2(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
    int    StrideF;
    float *ImgF1 = MallocPlaneFloat(Size.width, Size.height, &StrideF);

    CopyByte2Float(ImgSrc, Stride, ImgF1, StrideF, Size);
    AddFloatPlane(-128.0f, ImgF1, StrideF, Size);

    float *src, *dst;
    size_t DeviceStride;
    checkCudaErrors(cudaMallocPitch((void **)&src, &DeviceStride, Size.width * sizeof(float), Size.height));
    checkCudaErrors(cudaMallocPitch((void **)&dst, &DeviceStride, Size.width * sizeof(float), Size.height));
    DeviceStride /= sizeof(float);

    checkCudaErrors(cudaMemcpy2D(src,
                                 DeviceStride * sizeof(float),
                                 ImgF1,
                                 StrideF * sizeof(float),
                                 Size.width * sizeof(float),
                                 Size.height,
                                 cudaMemcpyHostToDevice));

    StopWatchInterface *timerCUDA = 0;
    sdkCreateTimer(&timerCUDA);

    dim3 GridFullWarps(Size.width / KER2_BLOCK_WIDTH, Size.height / KER2_BLOCK_HEIGHT, 1);
    dim3 ThreadsFullWarps(8, KER2_BLOCK_WIDTH / 8, KER2_BLOCK_HEIGHT / 8);

    const int numIterations = 100;

    for (int i = -1; i < numIterations; i++) {
        if (i == 0) {
            checkCudaErrors(cudaDeviceSynchronize());
            sdkResetTimer(&timerCUDA);
            sdkStartTimer(&timerCUDA);
        }

        CUDAkernel2DCT<<<GridFullWarps, ThreadsFullWarps>>>(dst, src, (int)DeviceStride);
        getLastCudaError("Kernel execution failed");
    }

    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timerCUDA);

    float avgTime = (float)sdkGetTimerValue(&timerCUDA) / (float)numIterations;
    sdkDeleteTimer(&timerCUDA);
    printf("%f MPix/s //%f ms\n", (1E-6 * (float)Size.width * (float)Size.height) / (1E-3 * avgTime), avgTime);

    dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    CUDAkernelQuantizationFloat<<<GridSmallBlocks, ThreadsSmallBlocks>>>(dst, (int)DeviceStride);
    getLastCudaError("Kernel execution failed");

    CUDAkernel2IDCT<<<GridFullWarps, ThreadsFullWarps>>>(src, dst, (int)DeviceStride);
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpy2D(ImgF1,
                                 StrideF * sizeof(float),
                                 src,
                                 DeviceStride * sizeof(float),
                                 Size.width * sizeof(float),
                                 Size.height,
                                 cudaMemcpyDeviceToHost));

    AddFloatPlane(128.0f, ImgF1, StrideF, Size);
    CopyFloat2Byte(ImgF1, StrideF, ImgDst, Stride, Size);

    checkCudaErrors(cudaFree(dst));
    checkCudaErrors(cudaFree(src));
    FreePlane(ImgF1);

    return avgTime;
}


float WrapperCUDAshort(byte *ImgSrc, byte *ImgDst, int Stride, ROI Size)
{
    int    StrideS;
    short *ImgS1 = MallocPlaneShort(Size.width, Size.height, &StrideS);

    for (int i = 0; i < Size.height; i++) {
        for (int j = 0; j < Size.width; j++) {
            ImgS1[i * StrideS + j] = (short)ImgSrc[i * Stride + j] - 128;
        }
    }

    short *SrcDst;
    size_t DeviceStride;
    checkCudaErrors(cudaMallocPitch((void **)(&SrcDst), &DeviceStride, Size.width * sizeof(short), Size.height));
    DeviceStride /= sizeof(short);

    checkCudaErrors(cudaMemcpy2D(SrcDst,
                                 DeviceStride * sizeof(short),
                                 ImgS1,
                                 StrideS * sizeof(short),
                                 Size.width * sizeof(short),
                                 Size.height,
                                 cudaMemcpyHostToDevice));

    StopWatchInterface *timerLibJpeg = 0;
    sdkCreateTimer(&timerLibJpeg);
    sdkResetTimer(&timerLibJpeg);

    dim3 GridShort(Size.width / KERS_BLOCK_WIDTH, Size.height / KERS_BLOCK_HEIGHT, 1);
    dim3 ThreadsShort(8, KERS_BLOCK_WIDTH / 8, KERS_BLOCK_HEIGHT / 8);

    sdkStartTimer(&timerLibJpeg);
    CUDAkernelShortDCT<<<GridShort, ThreadsShort>>>(SrcDst, (int)DeviceStride);
    checkCudaErrors(cudaDeviceSynchronize());
    sdkStopTimer(&timerLibJpeg);
    getLastCudaError("Kernel execution failed");

    float TimerLibJpegSpan16b = sdkGetAverageTimerValue(&timerLibJpeg);
    sdkDeleteTimer(&timerLibJpeg);

    dim3 ThreadsSmallBlocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 GridSmallBlocks(Size.width / BLOCK_SIZE, Size.height / BLOCK_SIZE);

    CUDAkernelQuantizationShort<<<GridSmallBlocks, ThreadsSmallBlocks>>>(SrcDst, (int)DeviceStride);
    getLastCudaError("Kernel execution failed");

    CUDAkernelShortIDCT<<<GridShort, ThreadsShort>>>(SrcDst, (int)DeviceStride);
    checkCudaErrors(cudaDeviceSynchronize());
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpy2D(ImgS1,
                                 StrideS * sizeof(short),
                                 SrcDst,
                                 DeviceStride * sizeof(short),
                                 Size.width * sizeof(short),
                                 Size.height,
                                 cudaMemcpyDeviceToHost));

    for (int i = 0; i < Size.height; i++) {
        for (int j = 0; j < Size.width; j++) {
            ImgDst[i * Stride + j] = clamp_0_255(ImgS1[i * StrideS + j] + 128);
        }
    }

    checkCudaErrors(cudaFree(SrcDst));
    FreePlane(ImgS1);

    return TimerLibJpegSpan16b;
}



int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    findCudaDevice(argc, (const char **)argv);

    char SampleImageFname[]             = "teapot512.bmp";
    char SampleImageFnameResGold1[]     = "teapot512_gold1.bmp";
    char SampleImageFnameResGold2[]     = "teapot512_gold2.bmp";
    char SampleImageFnameResCUDA1[]     = "teapot512_cuda1.bmp";
    char SampleImageFnameResCUDA2[]     = "teapot512_cuda2.bmp";
    char SampleImageFnameResCUDAshort[] = "teapot512_cuda_short.bmp";

    char *pSampleImageFpath = sdkFindFilePath(SampleImageFname, argv[0]);

    if (pSampleImageFpath == NULL) {
        printf("dct8x8 could not locate Sample Image <%s>\nExiting...\n", pSampleImageFpath);
        exit(EXIT_FAILURE);
    }

    int ImgWidth, ImgHeight;
    ROI ImgSize;
    int res        = PreLoadBmp(pSampleImageFpath, &ImgWidth, &ImgHeight);
    ImgSize.width  = ImgWidth;
    ImgSize.height = ImgHeight;

    printf("CUDA sample DCT/IDCT implementation\n");
    printf("===================================\n");
    printf("Loading test image: %s... ", SampleImageFname);

    if (res) {
        printf("\nError: Image file not found or invalid!\n");
        exit(EXIT_FAILURE);
        return 1;
    }

    if (ImgWidth % BLOCK_SIZE != 0 || ImgHeight % BLOCK_SIZE != 0) {
        printf("\nError: Input image dimensions must be multiples of 8!\n");
        exit(EXIT_FAILURE);
        return 1;
    }

    printf("[%d x %d]... ", ImgWidth, ImgHeight);

    int   ImgStride;
    byte *ImgSrc          = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstGold1     = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstGold2     = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstCUDA1     = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstCUDA2     = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);
    byte *ImgDstCUDAshort = MallocPlaneByte(ImgWidth, ImgHeight, &ImgStride);

    LoadBmpAsGray(pSampleImageFpath, ImgStride, ImgSize, ImgSrc);

    printf("Success\nRunning Gold 1 (CPU) version... ");
    float TimeGold1 = WrapperGold1(ImgSrc, ImgDstGold1, ImgStride, ImgSize);

    printf("Success\nRunning Gold 2 (CPU) version... ");
    float TimeGold2 = WrapperGold2(ImgSrc, ImgDstGold2, ImgStride, ImgSize);

    printf("Success\nRunning CUDA 1 (GPU) version... ");
    float TimeCUDA1 = WrapperCUDA1(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);

    printf("Success\nRunning CUDA 2 (GPU) version... ");
    float TimeCUDA2 = WrapperCUDA2(ImgSrc, ImgDstCUDA2, ImgStride, ImgSize);

    printf("Success\nRunning CUDA short (GPU) version... ");
    float TimeCUDAshort = WrapperCUDAshort(ImgSrc, ImgDstCUDAshort, ImgStride, ImgSize);

    printf("Success\nDumping result to %s... ", SampleImageFnameResGold1);
    DumpBmpAsGray(SampleImageFnameResGold1, ImgDstGold1, ImgStride, ImgSize);

    printf("Success\nDumping result to %s... ", SampleImageFnameResGold2);
    DumpBmpAsGray(SampleImageFnameResGold2, ImgDstGold2, ImgStride, ImgSize);

    printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA1);
    DumpBmpAsGray(SampleImageFnameResCUDA1, ImgDstCUDA1, ImgStride, ImgSize);

    printf("Success\nDumping result to %s... ", SampleImageFnameResCUDA2);
    DumpBmpAsGray(SampleImageFnameResCUDA2, ImgDstCUDA2, ImgStride, ImgSize);

    printf("Success\nDumping result to %s... ", SampleImageFnameResCUDAshort);
    DumpBmpAsGray(SampleImageFnameResCUDAshort, ImgDstCUDAshort, ImgStride, ImgSize);
    printf("Success\n");

    printf("Processing time (CUDA 1)    : %f ms \n", TimeCUDA1);
    printf("Processing time (CUDA 2)    : %f ms \n", TimeCUDA2);
    printf("Processing time (CUDA short): %f ms \n", TimeCUDAshort);

    float PSNR_Src_DstGold1        = CalculatePSNR(ImgSrc, ImgDstGold1, ImgStride, ImgSize);
    float PSNR_Src_DstGold2        = CalculatePSNR(ImgSrc, ImgDstGold2, ImgStride, ImgSize);
    float PSNR_Src_DstCUDA1        = CalculatePSNR(ImgSrc, ImgDstCUDA1, ImgStride, ImgSize);
    float PSNR_Src_DstCUDA2        = CalculatePSNR(ImgSrc, ImgDstCUDA2, ImgStride, ImgSize);
    float PSNR_Src_DstCUDAshort    = CalculatePSNR(ImgSrc, ImgDstCUDAshort, ImgStride, ImgSize);
    float PSNR_DstGold1_DstCUDA1   = CalculatePSNR(ImgDstGold1, ImgDstCUDA1, ImgStride, ImgSize);
    float PSNR_DstGold2_DstCUDA2   = CalculatePSNR(ImgDstGold2, ImgDstCUDA2, ImgStride, ImgSize);
    float PSNR_DstGold2_DstCUDA16b = CalculatePSNR(ImgDstGold2, ImgDstCUDAshort, ImgStride, ImgSize);

    printf("PSNR Original    <---> CPU(Gold 1)    : %f\n", PSNR_Src_DstGold1);
    printf("PSNR Original    <---> CPU(Gold 2)    : %f\n", PSNR_Src_DstGold2);
    printf("PSNR Original    <---> GPU(CUDA 1)    : %f\n", PSNR_Src_DstCUDA1);
    printf("PSNR Original    <---> GPU(CUDA 2)    : %f\n", PSNR_Src_DstCUDA2);
    printf("PSNR Original    <---> GPU(CUDA short): %f\n", PSNR_Src_DstCUDAshort);
    printf("PSNR CPU(Gold 1) <---> GPU(CUDA 1)    : %f\n", PSNR_DstGold1_DstCUDA1);
    printf("PSNR CPU(Gold 2) <---> GPU(CUDA 2)    : %f\n", PSNR_DstGold2_DstCUDA2);
    printf("PSNR CPU(Gold 2) <---> GPU(CUDA short): %f\n", PSNR_DstGold2_DstCUDA16b);

    bool bTestResult = (PSNR_DstGold1_DstCUDA1 > PSNR_THRESHOLD_EQUAL && PSNR_DstGold2_DstCUDA2 > PSNR_THRESHOLD_EQUAL
                        && PSNR_DstGold2_DstCUDA16b > PSNR_THRESHOLD_EQUAL);

    FreePlane(ImgSrc);
    FreePlane(ImgDstGold1);
    FreePlane(ImgDstGold2);
    FreePlane(ImgDstCUDA1);
    FreePlane(ImgDstCUDA2);
    FreePlane(ImgDstCUDAshort);

    printf("\nTest Summary...\n");

    if (!bTestResult) {
        printf("Test failed!\n");
        exit(EXIT_FAILURE);
    }

    printf("Test passed\n");
    exit(EXIT_SUCCESS);
}
