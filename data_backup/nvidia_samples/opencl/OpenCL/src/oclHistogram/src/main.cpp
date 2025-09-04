#include <oclUtils.h>
#include <shrQATest.h>
#include "oclHistogram_common.h"

int main(int argc, char **argv)
{
    cl_platform_id   cpPlatform;
    cl_device_id*    cdDevices;
    cl_context       cxGPUContext;
    cl_command_queue cqCommandQueue;
    cl_mem    d_Data, d_Histogram;
    cl_int ciErrNum;
    int PassFailFlag = 1;
    uchar *h_Data;
    uint *h_HistogramCPU, *h_HistogramGPU;
    uint byteCount = 64 * 1048576;
    uint uiSizeMult = 1;

    shrQAStart(argc, argv);

    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("clGetPlatformID...\n"); 

    cl_uint uiNumDevices = 0;
    cl_uint uiTargetDevice = 0;
    cl_uint uiNumComputeUnits;
    shrLog("Get the Device info and select Device...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);

    shrLog("  # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
    shrLog("  Using Device %u: ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);
    ciErrNum = clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, NULL);
    shrLog("\n  # of Compute Units = %u\n", uiNumComputeUnits); 

    shrSetLogFileName ("oclHistogram.txt");
    shrLog("%s Starting...\n\n", argv[0]); 

    if (shrGetCmdLineArgumentu(argc, (const char**)argv, "sizemult", &uiSizeMult))
    {
        uiSizeMult = CLAMP(uiSizeMult, 1, 10);
        byteCount *= uiSizeMult;
    }

    shrLog("Initializing data...\n");
        h_Data         = (uchar *)malloc(byteCount              * sizeof(uchar));
        h_HistogramCPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
        h_HistogramGPU = (uint  *)malloc(HISTOGRAM256_BIN_COUNT * sizeof(uint));
        srand(2009);
        for(uint i = 0; i < byteCount; i++)
            h_Data[i] = rand() & 0xFFU;

    shrLog("Initializing OpenCL...\n");
        ciErrNum = oclGetPlatformID(&cpPlatform);
        oclCheckError(ciErrNum, CL_SUCCESS);

        ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &cdDevices[uiTargetDevice], NULL);
        oclCheckError(ciErrNum, CL_SUCCESS);

        cxGPUContext = clCreateContext(0, 1, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

        cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiTargetDevice], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);

    shrLog("Allocating OpenCL memory...\n\n\n");
        d_Data = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, byteCount * sizeof(cl_char), h_Data, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);
        d_Histogram = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, HISTOGRAM256_BIN_COUNT * sizeof(uint), NULL, &ciErrNum);
        oclCheckError(ciErrNum, CL_SUCCESS);


    {
        size_t szWorkgroup;
        shrLog("Initializing 64-bin OpenCL histogram...\n");
            initHistogram64(cxGPUContext, cqCommandQueue, (const char **)argv);

        shrLog("Running 64-bin OpenCL histogram for %u bytes...\n\n", byteCount);
            szWorkgroup = histogram64(NULL, d_Histogram, d_Data, byteCount);

#ifdef GPU_PROFILING
        const uint numIterations = 16;
        cl_event startMark, endMark;
        ciErrNum = clEnqueueMarker(cqCommandQueue, &startMark);
        ciErrNum |= clFinish(cqCommandQueue);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrDeltaT(0);

        for(uint iter = 0; iter < numIterations; iter++)
            szWorkgroup = histogram64(NULL, d_Histogram, d_Data, byteCount);

        ciErrNum  = clEnqueueMarker(cqCommandQueue, &endMark);
        ciErrNum |= clFinish(cqCommandQueue);
        shrCheckError(ciErrNum, CL_SUCCESS);

        double gpuTime = shrDeltaT(0) / (double)numIterations;
        shrLogEx(LOGBOTH | MASTER, 0, "oclHistogram64, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n", 
                (1.0e-6 * (double)byteCount / gpuTime), gpuTime, byteCount, 1, szWorkgroup); 

        cl_ulong startTime = 0, endTime = 0;
        ciErrNum  = clGetEventProfilingInfo(startMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &startTime, NULL);
        ciErrNum |= clGetEventProfilingInfo(endMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog("\nOpenCL time: %.5f s\n\n", 1.0e-9 * ((double)endTime - (double)startTime)/(double)numIterations);
#endif

        shrLog("Validating 64-bin histogram OpenCL results...\n");
            shrLog(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Histogram, CL_TRUE, 0, HISTOGRAM64_BIN_COUNT * sizeof(uint), h_HistogramGPU, 0, NULL, NULL);
                shrCheckError(ciErrNum, CL_SUCCESS);

            shrLog(" ...histogram64CPU()\n");
                histogram64CPU(h_HistogramCPU, h_Data, byteCount);

            shrLog(" ...comparing the results\n");
                for(uint i = 0; i < HISTOGRAM64_BIN_COUNT; i++)
                    if(h_HistogramGPU[i] != h_HistogramCPU[i]) PassFailFlag = 0;
            shrLog(PassFailFlag ? " ...64-bin histograms match\n\n" : " ***64-bin histograms do not match!!!***\n\n" );

        shrLog("Shutting down 64-bin OpenCL histogram\n\n\n"); 
            closeHistogram64();
    }

    {
        size_t szWorkgroup;
        shrLog("Initializing 256-bin OpenCL histogram...\n");
            initHistogram256(cxGPUContext, cqCommandQueue, (const char **)argv);

        shrLog("Running 256-bin OpenCL histogram for %u bytes...\n\n", byteCount);
            szWorkgroup = histogram256(NULL, d_Histogram, d_Data, byteCount);

#ifdef GPU_PROFILING
        const uint numIterations = 16;
        cl_event startMark, endMark;
        ciErrNum = clEnqueueMarker(cqCommandQueue, &startMark);
        ciErrNum |= clFinish(cqCommandQueue);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrDeltaT(0);

        for(uint iter = 0; iter < numIterations; iter++)
            szWorkgroup = histogram256(NULL, d_Histogram, d_Data, byteCount);

        ciErrNum  = clEnqueueMarker(cqCommandQueue, &endMark);
        ciErrNum |= clFinish(cqCommandQueue);
        shrCheckError(ciErrNum, CL_SUCCESS);

        double gpuTime = shrDeltaT(0) / (double)numIterations;
        shrLogEx(LOGBOTH | MASTER, 0, "oclHistogram256, Throughput = %.4f MB/s, Time = %.5f s, Size = %u Bytes, NumDevsUsed = %u, Workgroup = %u\n", 
                (1.0e-6 * (double)byteCount / gpuTime), gpuTime, byteCount, 1, szWorkgroup); 

        cl_ulong startTime = 0, endTime = 0;
        ciErrNum  = clGetEventProfilingInfo(startMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &startTime, NULL);
        ciErrNum |= clGetEventProfilingInfo(endMark, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &endTime, NULL);
        shrCheckError(ciErrNum, CL_SUCCESS);
        shrLog("\nOpenCL time: %.5f s\n\n", 1.0e-9 * (double)(endTime - startTime)/(double)numIterations);
#endif

        shrLog("Validating 256-bin histogram OpenCL results...\n");
            shrLog(" ...reading back OpenCL results\n");
                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, d_Histogram, CL_TRUE, 0, HISTOGRAM256_BIN_COUNT * sizeof(uint), h_HistogramGPU, 0, NULL, NULL);
                shrCheckError(ciErrNum, CL_SUCCESS);

            shrLog(" ...histogram256CPU()\n");
                histogram256CPU(h_HistogramCPU, h_Data, byteCount);

            shrLog(" ...comparing the results\n");
                for(uint i = 0; i < HISTOGRAM256_BIN_COUNT; i++)
                    if(h_HistogramGPU[i] != h_HistogramCPU[i]) PassFailFlag = 0;
            shrLog(PassFailFlag ? " ...256-bin histograms match\n\n" : " ***256-bin histograms do not match!!!***\n\n" );

        shrLog("Shutting down 256-bin OpenCL histogram\n\n\n"); 
            closeHistogram256();
    }

    shrLog("Shutting down...\n");
        ciErrNum  = clReleaseMemObject(d_Histogram);
        ciErrNum |= clReleaseMemObject(d_Data);
        ciErrNum |= clReleaseCommandQueue(cqCommandQueue);
        ciErrNum |= clReleaseContext(cxGPUContext);
        oclCheckError(ciErrNum, CL_SUCCESS);

        free(h_HistogramGPU);
        free(h_HistogramCPU);
        free(h_Data);

        if (cdDevices) free(cdDevices);

    shrQAFinishExit(argc, (const char **)argv, PassFailFlag ? QA_PASSED : QA_FAILED);
}
