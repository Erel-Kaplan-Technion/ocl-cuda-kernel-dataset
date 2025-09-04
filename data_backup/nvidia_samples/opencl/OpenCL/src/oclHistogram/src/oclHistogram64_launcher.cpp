#include <oclUtils.h>
#include "oclHistogram_common.h"

static cl_program cpHistogram64;

static cl_kernel
    ckHistogram64,
    ckMergeHistogram64;

static cl_command_queue cqDefaultCommandQue;

static const uint MAX_PARTIAL_HISTOGRAM64_COUNT = 32768;
static cl_mem d_PartialHistograms;

static const uint HISTOGRAM64_WORKGROUP_SIZE = 64;
static const uint       MERGE_WORKGROUP_SIZE = 256;
static const char            *compileOptions = "-D LOCAL_MEMORY_BANKS=16 -D HISTOGRAM64_WORKGROUP_SIZE=64 -D MERGE_WORKGROUP_SIZE=256";

extern "C" void initHistogram64(cl_context cxGPUContext, cl_command_queue cqParamCommandQue, const char **argv){
    cl_int ciErrNum;
    size_t kernelLength;

    shrLog("...loading Histogram64.cl from file\n");
        char *cHistogram64 = oclLoadProgSource(shrFindFilePath("Histogram64.cl", argv[0]), "// My comment\n", &kernelLength);
        shrCheckError(cHistogram64 != NULL, shrTRUE);

    shrLog("...creating histogram64 program\n");
         cpHistogram64 = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cHistogram64, &kernelLength, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog("...building histogram64 program\n");
        ciErrNum = clBuildProgram(cpHistogram64, 0, NULL, compileOptions, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog("...creating histogram64 kernels\n");
        ckHistogram64 = clCreateKernel(cpHistogram64, "histogram64", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);
        ckMergeHistogram64 = clCreateKernel(cpHistogram64, "mergeHistogram64", &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    shrLog("...allocating internal histogram64 buffer\n");
        d_PartialHistograms = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, MAX_PARTIAL_HISTOGRAM64_COUNT * HISTOGRAM64_BIN_COUNT * sizeof(uint), NULL, &ciErrNum);
        shrCheckError(ciErrNum, CL_SUCCESS);

    cqDefaultCommandQue = cqParamCommandQue;

    free(cHistogram64);

    //Save ptx code to separate file
    oclLogPtx(cpHistogram64, oclGetFirstDev(cxGPUContext), "Histogram64.ptx");
}

extern "C" void closeHistogram64(void){
    cl_int ciErrNum;

    ciErrNum  = clReleaseMemObject(d_PartialHistograms);
    ciErrNum |= clReleaseKernel(ckMergeHistogram64);
    ciErrNum |= clReleaseKernel(ckHistogram64);
    ciErrNum |= clReleaseProgram(cpHistogram64);
    shrCheckError(ciErrNum, CL_SUCCESS);
}

inline uint iDivUp(uint a, uint b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline uint iSnapDown(uint a, uint b){
    return a - a % b;
}

extern "C" size_t histogram64(
    cl_command_queue cqCommandQueue,
    cl_mem d_Histogram,
    cl_mem d_Data,
    uint byteCount
){
    cl_int ciErrNum;
    uint histogramCount;
    size_t localWorkSize, globalWorkSize;

    if(!cqCommandQueue)
        cqCommandQueue = cqDefaultCommandQue;

    {
        histogramCount = iDivUp(byteCount, HISTOGRAM64_WORKGROUP_SIZE * iSnapDown(255, 16));
        shrCheckError( (byteCount % 16 == 0), shrTRUE );
        shrCheckError( (histogramCount <= MAX_PARTIAL_HISTOGRAM64_COUNT), shrTRUE );
        cl_uint dataCount = byteCount / 16;

        ciErrNum  = clSetKernelArg(ckHistogram64, 0, sizeof(cl_mem),  (void *)&d_PartialHistograms);
        ciErrNum |= clSetKernelArg(ckHistogram64, 1, sizeof(cl_mem),  (void *)&d_Data);
        ciErrNum |= clSetKernelArg(ckHistogram64, 2, sizeof(cl_uint), (void *)&dataCount);
        shrCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize = HISTOGRAM64_WORKGROUP_SIZE;
        globalWorkSize = histogramCount * HISTOGRAM64_WORKGROUP_SIZE;

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckHistogram64, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);
    }

    {
        ciErrNum  = clSetKernelArg(ckMergeHistogram64, 0, sizeof(cl_mem),  (void *)&d_Histogram);
        ciErrNum |= clSetKernelArg(ckMergeHistogram64, 1, sizeof(cl_mem),  (void *)&d_PartialHistograms);
        ciErrNum |= clSetKernelArg(ckMergeHistogram64, 2, sizeof(cl_uint), (void *)&histogramCount);
        shrCheckError(ciErrNum, CL_SUCCESS);

        localWorkSize = MERGE_WORKGROUP_SIZE;
        globalWorkSize = HISTOGRAM64_BIN_COUNT * MERGE_WORKGROUP_SIZE;

        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckMergeHistogram64, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);

        return HISTOGRAM64_WORKGROUP_SIZE;
    }
}
