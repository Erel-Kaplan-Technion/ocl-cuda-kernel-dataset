#include <GL/glew.h>
#ifdef UNIX
    #include <GL/glxew.h>
#endif
#if defined (_WIN32)
    #include <GL/wglew.h>
#endif

#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
#endif

#include "DeviceManager.h"

#include <oclUtils.h>
#include <shrQATest.h>

#ifndef min
#define min(a,b) (a < b ? a : b)
#endif

int *pArgc = NULL;
char **pArgv = NULL;

extern "C" double SobelFilterHost(unsigned int* uiInputImage, unsigned int* uiOutputImage, 
                                  unsigned int uiWidth, unsigned int uiHeight, float fThresh);

#define REFRESH_DELAY	  10 //ms

int iBlockDimX = 16;
int iBlockDimY = 4;
float fThresh = 80.0f;

const char* cImageFile = "StoneRGB.ppm";
unsigned int uiImageWidth = 1920;
unsigned int uiImageHeight = 1080;

int iGLUTWindowHandle;
int iGLUTMenuHandle;
int iGraphicsWinPosX = 0;
int iGraphicsWinPosY = 0;
int iGraphicsWinWidth = 1024;
int iGraphicsWinHeight = ((float)uiImageHeight / (float)uiImageWidth) * iGraphicsWinWidth;
float fZoom = 1.0f;
int iFrameCount = 0;
int iFrameTrigger = 90;
int iFramesPerSec = 60;
double dProcessingTime = 0.0;
bool bFullScreen = false;
GLint iVsyncState;

const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
bool bFilter = true;
int iProcFlag = 0;
bool bNoPrompt = shrFALSE;
bool bQATest = shrFALSE;
int iTestSets = 3;

const char* clSourcefile = "SobelFilter.cl";
char* cPathAndName = NULL;
char* cSourceCL = NULL;
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_command_queue* cqCommandQueue;
cl_uint* uiInHostPixOffsets;
cl_uint* uiOutHostPixOffsets;
cl_program cpProgram;
cl_kernel* ckSobel;
cl_mem cmPinnedBufIn;
cl_mem cmPinnedBufOut;
cl_mem* cmDevBufIn;
cl_mem* cmDevBufOut;
cl_uint* uiInput = NULL;
cl_uint* uiOutput = NULL;
size_t szBuffBytes;
size_t* szAllocDevBytes;
cl_uint* uiDevImageHeight;
size_t szGlobalWorkSize[2];
size_t szLocalWorkSize[2];
size_t szParmDataBytes;
size_t szKernelLength;
cl_int ciErrNum;
const char* cExecutableName = NULL;

DeviceManager* GpuDevMngr; 

double SobelFilterGPU(unsigned int* uiInputImage, unsigned int* uiOutputImage);

void InitGL(int* argc, char** argv);
void DeInitGL();
void DisplayGL();
void Reshape(int w, int h);
void Idle(void);
void KeyboardGL(unsigned char key, int x, int y);
void MenuGL(int i);
void timerEvent(int value);

void TestNoGL();
void TriggerFPSUpdate();
void GetDeviceLoadProportions(float* fLoadProportions, cl_device_id* cdDevices, cl_uint uiDevCount);
void ShowMenuItems();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

int main(int argc, char** argv)
{
	pArgc = &argc;
	pArgv = argv;

	shrQAStart(argc, argv);

	cExecutableName = argv[0];
    shrSetLogFileName ("oclSobelFilter.txt");
    shrLog("%s Starting (Using %s)...\n\n", argv[0], clSourcefile); 

    bNoPrompt = (bool)shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    bQATest   = (bool)shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");

    if (!(bQATest))
    {
        ShowMenuItems();
    }

    cPathAndName = shrFindFilePath(cImageFile, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    shrLog("Image File\t = %s\nImage Dimensions = %u w x %u h x %u bpp\n\n", cPathAndName, uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3);

    shrLog("%sInitGL...\n\n", bQATest ? "Skipping " : "Calling "); 
    if (!(bQATest))
    {
        InitGL(&argc, argv);
    }

    char cBuffer[1024];
    bool bNV = false;
    shrLog("Get Platform ID... ");
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    ciErrNum = clGetPlatformInfo (cpPlatform, CL_PLATFORM_NAME, sizeof(cBuffer), cBuffer, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("%s\n\n", cBuffer);
    bNV = (strstr(cBuffer, "NVIDIA") != NULL);

    shrLog("Get Device Info...\n");
    cl_uint uiNumAllDevs = 0;
    GpuDevMngr = new DeviceManager(cpPlatform, &uiNumAllDevs, pCleanup);

    cl_int iSelectedDevice = 0;
    if((shrGetCmdLineArgumenti(argc, (const char**)argv, "device", &iSelectedDevice)) || (uiNumAllDevs == 1)) 
    {
        GpuDevMngr->uiUsefulDevCt = 1;  
        iSelectedDevice = CLAMP((cl_uint)iSelectedDevice, 0, (uiNumAllDevs - 1));
        GpuDevMngr->uiUsefulDevs[0] = iSelectedDevice;
        GpuDevMngr->fLoadProportions[0] = 1.0f;
        shrLog("  Using 1 Selected Device for Sobel Filter Computation...\n"); 
 
    } 
    else 
    {
        ciErrNum = GpuDevMngr->GetDevLoadProportions(bNV);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        if (GpuDevMngr->uiUsefulDevCt == 1)
        {
            iSelectedDevice = GpuDevMngr->uiUsefulDevs[0];
        }
        shrLog("    Using %u Device(s) for Sobel Filter Computation\n", GpuDevMngr->uiUsefulDevCt); 
    }

    shrLog("\nclCreateContext...\n\n");
    cxGPUContext = clCreateContext(0, uiNumAllDevs, GpuDevMngr->cdDevices, NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    cqCommandQueue = new cl_command_queue[GpuDevMngr->uiUsefulDevCt];
    ckSobel = new cl_kernel[GpuDevMngr->uiUsefulDevCt];
    cmDevBufIn = new cl_mem[GpuDevMngr->uiUsefulDevCt];
    cmDevBufOut = new cl_mem[GpuDevMngr->uiUsefulDevCt];
    szAllocDevBytes = new size_t[GpuDevMngr->uiUsefulDevCt];
    uiInHostPixOffsets = new cl_uint[GpuDevMngr->uiUsefulDevCt];
    uiOutHostPixOffsets = new cl_uint[GpuDevMngr->uiUsefulDevCt];
    uiDevImageHeight = new cl_uint[GpuDevMngr->uiUsefulDevCt];

    shrLog("clCreateCommandQueue...\n");
    for (cl_uint i = 0; i < GpuDevMngr->uiUsefulDevCt; i++) 
    {
        cqCommandQueue[i] = clCreateCommandQueue(cxGPUContext, GpuDevMngr->cdDevices[GpuDevMngr->uiUsefulDevs[i]], 0, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("  CommandQueue %u, Device %u, Device Load Proportion = %.2f, ", i, GpuDevMngr->uiUsefulDevs[i], GpuDevMngr->fLoadProportions[i]); 
        oclPrintDevName(LOGBOTH, GpuDevMngr->cdDevices[GpuDevMngr->uiUsefulDevs[i]]);  
        shrLog("\n");
    }

    szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
    cmPinnedBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmPinnedBufOut = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("\nclCreateBuffer (Input and Output Pinned Host buffers)...\n"); 

    uiInput = (cl_uint*)clEnqueueMapBuffer(cqCommandQueue[0], cmPinnedBufIn, CL_TRUE, CL_MAP_WRITE, 0, szBuffBytes, 0, NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    uiOutput = (cl_uint*)clEnqueueMapBuffer(cqCommandQueue[0], cmPinnedBufOut, CL_TRUE, CL_MAP_READ, 0, szBuffBytes, 0, NULL, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clEnqueueMapBuffer (Pointer to Input and Output pinned host buffers)...\n"); 

    ciErrNum = shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
    oclCheckErrorEX(ciErrNum, shrTRUE, pCleanup);
    shrLog("Load Input Image to Input pinned host buffer...\n"); 

    free(cPathAndName);
    cPathAndName = shrFindFilePath(clSourcefile, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "// My comment\n", &szKernelLength);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);
    shrLog("Load OpenCL Prog Source from File...\n"); 

    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateProgramWithSource...\n"); 

#ifdef MAC
    char *flags = "-cl-fast-relaxed-math -DMAC";
#else
    char *flags = "-cl-fast-relaxed-math";
#endif

    ciErrNum = clBuildProgram(cpProgram, 0, NULL, flags, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "oclSobelFilter.ptx");
        Cleanup(EXIT_FAILURE);
    }
    shrLog("clBuildProgram...\n\n"); 

    unsigned uiSumHeight = 0;
    for (cl_uint i = 0; i < GpuDevMngr->uiUsefulDevCt; i++)
    {
        ckSobel[i] = clCreateKernel(cpProgram, "ckSobel", &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clCreateKernel (ckSobel), Device %u...\n", i); 

        if (GpuDevMngr->uiUsefulDevCt == 1)
        {
            uiDevImageHeight[i] = uiImageHeight; 
            uiInHostPixOffsets[i] = 0;
            uiOutHostPixOffsets[i] = 0;
            szAllocDevBytes[i] = uiDevImageHeight[i] * uiImageWidth * sizeof(cl_uint);
        }
        else if (i == 0)
        {
            uiInHostPixOffsets[i] = 0;
            uiOutHostPixOffsets[i] = 0;
            uiDevImageHeight[i] = (cl_uint)(GpuDevMngr->fLoadProportions[GpuDevMngr->uiUsefulDevs[i]] * (float)uiImageHeight);     // height is proportional to dev perf 
            uiSumHeight += uiDevImageHeight[i];
            uiDevImageHeight[i] += 1;
            szAllocDevBytes[i] = uiDevImageHeight[i] * uiImageWidth * sizeof(cl_uint);
        }
        else if (i < (GpuDevMngr->uiUsefulDevCt - 1))
        {
            uiInHostPixOffsets[i] = (uiSumHeight - 1) * uiImageWidth;
            uiOutHostPixOffsets[i] = uiInHostPixOffsets[i] + uiImageWidth;
            uiDevImageHeight[i] = (cl_uint)(GpuDevMngr->fLoadProportions[GpuDevMngr->uiUsefulDevs[i]] * (float)uiImageHeight);     // height is proportional to dev perf 
            uiSumHeight += uiDevImageHeight[i];
            uiDevImageHeight[i] += 2;
            szAllocDevBytes[i] = uiDevImageHeight[i] * uiImageWidth * sizeof(cl_uint);
        }
        else 
        {
            uiInHostPixOffsets[i] = (uiSumHeight - 1) * uiImageWidth;
            uiOutHostPixOffsets[i] = uiInHostPixOffsets[i] + uiImageWidth;
            uiDevImageHeight[i] = uiImageHeight - uiSumHeight;                              // "leftover" rows 
            uiSumHeight += uiDevImageHeight[i];
            uiDevImageHeight[i] += 1;
            szAllocDevBytes[i] = uiDevImageHeight[i] * uiImageWidth * sizeof(cl_uint);
        }
        shrLog("Image Height (rows) for Device %u = %u...\n", i, uiDevImageHeight[i]); 

        cmDevBufIn[i] = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, szAllocDevBytes[i], NULL, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        cmDevBufOut[i] = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, szAllocDevBytes[i], NULL, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clCreateBuffer (Input and Output GMEM buffers, Device %u)...\n", i); 

        int iLocalPixPitch = iBlockDimX + 2;
        ciErrNum = clSetKernelArg(ckSobel[i], 0, sizeof(cl_mem), (void*)&cmDevBufIn[i]);
        ciErrNum |= clSetKernelArg(ckSobel[i], 1, sizeof(cl_mem), (void*)&cmDevBufOut[i]);
        ciErrNum |= clSetKernelArg(ckSobel[i], 2, (iLocalPixPitch * (iBlockDimY + 2) * sizeof(cl_uchar4)), NULL);
        ciErrNum |= clSetKernelArg(ckSobel[i], 3, sizeof(cl_int), (void*)&iLocalPixPitch);
        ciErrNum |= clSetKernelArg(ckSobel[i], 4, sizeof(cl_uint), (void*)&uiImageWidth);
        ciErrNum |= clSetKernelArg(ckSobel[i], 6, sizeof(cl_float), (void*)&fThresh);        
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clSetKernelArg (0-4), Device %u...\n\n", i); 
    }

    szLocalWorkSize[0] = iBlockDimX;
    szLocalWorkSize[1] = iBlockDimY;
    szGlobalWorkSize[0] = shrRoundUp((int)szLocalWorkSize[0], uiImageWidth); 

    shrDeltaT(0);
    shrDeltaT(1);

    if (!(bQATest))
    {
        glutMainLoop();
    }
    else 
    {
        TestNoGL();
    }

    Cleanup(EXIT_SUCCESS);
}

double SobelFilterGPU(cl_uint* uiInputImage, cl_uint* uiOutputImage)
{
    ciErrNum = CL_SUCCESS;
    for (cl_uint i = 0; i < GpuDevMngr->uiUsefulDevCt; i++)
    {
        ciErrNum |= clEnqueueWriteBuffer(cqCommandQueue[i], cmDevBufIn[i], CL_FALSE, 0, szAllocDevBytes[i], 
                                        (void*)&uiInputImage[uiInHostPixOffsets[i]], 0, NULL, NULL);
    }

    for (cl_uint j = 0; j < GpuDevMngr->uiUsefulDevCt; j++)
    {
        ciErrNum |= clFinish(cqCommandQueue[j]);
    }
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    shrDeltaT(0);
    for (cl_uint i = 0; i < GpuDevMngr->uiUsefulDevCt; i++)
    {
        if (GpuDevMngr->uiUsefulDevCt == 1)
        {
            szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], (int)uiDevImageHeight[i]);
        }
        else if (i == 0)
        {
            szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], (int)uiDevImageHeight[i]);
        }
        else if (i < (GpuDevMngr->uiUsefulDevCt - 1))
        {
            szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], (int)uiDevImageHeight[i]);
        }
        else 
        {   
            szGlobalWorkSize[1] = shrRoundUp((int)szLocalWorkSize[1], (int)uiDevImageHeight[i]);
        }

        ciErrNum |= clSetKernelArg(ckSobel[i], 5, sizeof(cl_uint), (void*)&uiDevImageHeight[i]);

        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue[i], ckSobel[i], 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);

        ciErrNum |= clFlush(cqCommandQueue[i]);
    }

    for (cl_uint j = 0; j < GpuDevMngr->uiUsefulDevCt; j++)
    {
        ciErrNum |= clFinish(cqCommandQueue[j]);
    }
    double dKernelTime = shrDeltaT(0);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    for (cl_uint i = 0; i < GpuDevMngr->uiUsefulDevCt; i++)
    {
        size_t szReturnBytes;
        cl_uint uiOutDevByteOffset;        
        if (GpuDevMngr->uiUsefulDevCt == 1)
        {
            szReturnBytes = szBuffBytes;
            uiOutDevByteOffset = 0;
        } 
        else if (i == 0)
        {
            szReturnBytes = szAllocDevBytes[i] - (uiImageWidth * sizeof(cl_uint));
            uiOutDevByteOffset = 0;
        }
        else if (i < (GpuDevMngr->uiUsefulDevCt - 1))
        {
            szReturnBytes = szAllocDevBytes[i] - ((uiImageWidth * sizeof(cl_uint)) * 2);
            uiOutDevByteOffset = uiImageWidth * sizeof(cl_uint);
        }        
        else 
        {   
            szReturnBytes = szAllocDevBytes[i] - (uiImageWidth * sizeof(cl_uint));
            uiOutDevByteOffset = uiImageWidth * sizeof(cl_uint);
        }        
        
        ciErrNum |= clEnqueueReadBuffer(cqCommandQueue[i], cmDevBufOut[i], CL_FALSE, uiOutDevByteOffset, szReturnBytes, 
                                       (void*)&uiOutputImage[uiOutHostPixOffsets[i]], 0, NULL, NULL);
    }

    for (cl_uint j = 0; j < GpuDevMngr->uiUsefulDevCt; j++)
    {
        ciErrNum |= clFinish(cqCommandQueue[j]);
    }
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    return dKernelTime;
}

void InitGL(int* argc, char **argv)
{
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL for GPU RGB Sobel Filter Demo");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    glutKeyboardFunc(KeyboardGL);
    glutDisplayFunc(DisplayGL);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    iGLUTMenuHandle = glutCreateMenu(MenuGL);
    glutAddMenuEntry("Toggle Filter On/Off <spacebar>", ' ');
    glutAddMenuEntry("Toggle Processing between GPU and CPU [p]", 'p');
    glutAddMenuEntry("Toggle between Full Screen and Windowed [f]", 'f');
    glutAddMenuEntry("Increase Threshold [+]", '+');
    glutAddMenuEntry("Decrease Threshold [-]", '-');
    glutAddMenuEntry("Quit <esc>", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    glClearColor(0.f, 0.f, 0.f, 0.f);

    float fAspects[2] = {(float)glutGet(GLUT_WINDOW_WIDTH)/(float)uiImageWidth , (float)glutGet(GLUT_WINDOW_HEIGHT)/(float)uiImageHeight};
    fZoom = fAspects[0] > fAspects[1] ? fAspects[1] : fAspects[0];
    glPixelZoom(fZoom, fZoom);

    glewInit();

    #ifdef _WIN32
        if (wglewIsSupported("WGL_EXT_swap_control")) 
        {
            iVsyncState = wglGetSwapIntervalEXT();
            wglSwapIntervalEXT(0);
        }
    #else
        #if defined (__APPLE__) || defined(MACOSX)
            GLint VBL = 0;
            CGLGetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &iVsyncState); 
            CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &VBL); 
        #else        
            if(glxewIsSupported("GLX_SGI_swap_control"))
            {
                glXSwapIntervalSGI(0);	 
            }
        #endif
    #endif
}

void DeInitGL()
{
    #ifdef _WIN32
        if (wglewIsSupported("WGL_EXT_swap_control")) 
        {
            wglSwapIntervalEXT(iVsyncState);
        }
    #else
        #if defined (__APPLE__) || defined(MACOSX)
            CGLSetParameter(CGLGetCurrentContext(), kCGLCPSwapInterval, &iVsyncState); 
        #endif
    #endif

}

void DisplayGL()
{
    glClear(GL_COLOR_BUFFER_BIT);
    if (bFilter)
    {
        if (iProcFlag == 0)
        {
            dProcessingTime += SobelFilterGPU (uiInput, uiOutput);
        }
        else 
        {
            dProcessingTime += SobelFilterHost (uiInput, uiOutput, uiImageWidth, uiImageHeight, fThresh);
        }

        glDrawPixels(uiImageWidth, uiImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, uiOutput); 
    }
    else 
    {
        glDrawPixels(uiImageWidth, uiImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, uiInput); 
    }

    glutSwapBuffers();

    if (iFrameCount++ > iFrameTrigger)
    {
        char
        iFramesPerSec = (int)((double)iFrameCount / shrDeltaT(1));
        dProcessingTime /= (double)iFrameCount; 
        
#ifdef GPU_PROFILING
        if (bFilter)
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s RGB Sobel Filter ON | W: %u , H: %u | Thresh. = %.1f | # GPUs = %u | %i fps | Proc. t = %.5f s | %.1f Mpix/s", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fThresh, GpuDevMngr->uiUsefulDevCt, iFramesPerSec, 
                            dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
            #else
                sprintf(cTitle, "%s RGB Sobel Filter ON | W: %u , H: %u | Thresh. = %.1f | # GPUs = %u | %i fps | Proc. t = %.5f s | %.1f Mpix/s", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fThresh, GpuDevMngr->uiUsefulDevCt, iFramesPerSec, 
                            dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
            #endif
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "RGB Sobel Filter OFF | W: %u , H: %u | %i fps", 
                            uiImageWidth, uiImageHeight, iFramesPerSec);  
            #else 
                sprintf(cTitle, "RGB Sobel Filter OFF | W: %u , H: %u | %i fps", 
                            uiImageWidth, uiImageHeight, iFramesPerSec);  
            #endif
        }
#else
        if (bFilter)
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "%s RGB Sobel Filter ON | W: %u , H: %u | Thresh. = %.1f", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fThresh);  
            #else
                sprintf(cTitle, "%s RGB Sobel Filter ON | W: %u , H: %u | Thresh. = %.1f", 
                            cProcessor[iProcFlag], uiImageWidth, uiImageHeight, fThresh);  
            #endif
        }
        else 
        {
            #ifdef _WIN32
                sprintf_s(cTitle, 256, "RGB Sobel Filter OFF | W: %u , H: %u", 
                            uiImageWidth, uiImageHeight);  
            #else 
                sprintf(cTitle, "RGB Sobel Filter OFF | W: %u , H: %u", 
                            uiImageWidth, uiImageHeight);  
            #endif
        }
#endif
        glutSetWindowTitle(cTitle);

        shrLog("%s\n", cTitle); 

        if ((bNoPrompt) && (!--iTestSets))
        {
            Cleanup(EXIT_SUCCESS);
        }

        iFrameCount = 0; 
        dProcessingTime = 0.0;
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

void Reshape(int w, int h)
{
    glPixelZoom((float)w/uiImageWidth , (float)h/uiImageHeight);
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

void KeyboardGL(unsigned char key, int /*x*/, int /*y*/)
{
    switch(key) 
    {
        case 'P':
        case 'p':
            if (iProcFlag == 0)
            {
                iProcFlag = 1;
            }
            else 
            {
                iProcFlag = 0;
            }
            shrLog("\n%s Processing...\n", cProcessor[iProcFlag]);
            break;
        case 'F':
        case 'f':
            bFullScreen = !bFullScreen;
            if (bFullScreen)
            {
                iGraphicsWinPosX = glutGet(GLUT_WINDOW_X) - 8;
                iGraphicsWinPosY = glutGet(GLUT_WINDOW_Y) - 30;
                iGraphicsWinWidth  = min(glutGet(GLUT_WINDOW_WIDTH) , glutGet(GLUT_SCREEN_WIDTH) - 2*iGraphicsWinPosX ); 
                iGraphicsWinHeight = min(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_SCREEN_HEIGHT)- 2*iGraphicsWinPosY ); 
                printf("(x,y)=(%d,%d), (w,h)=(%d,%d)\n", iGraphicsWinPosX, iGraphicsWinPosY, iGraphicsWinWidth, iGraphicsWinHeight);
                glutFullScreen();
            }
            else
            {
                glutPositionWindow(iGraphicsWinPosX, iGraphicsWinPosY);
                glutReshapeWindow(iGraphicsWinWidth, iGraphicsWinHeight);
            }
            shrLog("\nMain Graphics %s...\n", bFullScreen ? "FullScreen" : "Windowed");
            break;
        case ' ':
            bFilter = !bFilter;
            shrLog("\nSobel Filter Toggled %s...\n", bFilter ? "ON" : "OFF");
            break;
        case '+':
        case '=':
        case '-':
        case '_':
            if(key == '+' || key == '=')
            {
                fThresh += 10.0f;
            }
            else
            {
                fThresh -= 10.0f;
            }

            fThresh = CLAMP(fThresh, 0.0f, 255.0f);
            for (cl_uint i = 0; i < GpuDevMngr->uiUsefulDevCt; i++)
            {
                ciErrNum = clSetKernelArg(ckSobel[i], 6, sizeof(cl_float), (void*)&fThresh);
            }
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            shrLog("\nThreshold changed to %.1f...\n", fThresh);
            break;
        case '\033':
        case '\015':
        case 'Q':
        case 'q':
            bNoPrompt = shrTRUE;
            Cleanup(EXIT_SUCCESS);
            break;
    }

    TriggerFPSUpdate();
}

void MenuGL(int i)
{
    KeyboardGL((unsigned char) i, 0, 0);
}

void Idle(void)
{
}

void TriggerFPSUpdate()
{
    iFrameCount = 0; 
    iFramesPerSec = 1;
    iFrameTrigger = 2;
    shrDeltaT(1);
    shrDeltaT(0);
    dProcessingTime = 0.0;
}

void TestNoGL()
{
    SobelFilterGPU (uiInput, uiOutput);

    const int iCycles = 150;
    dProcessingTime = 0.0;
    shrLog("\nRunning SobelFilterGPU for %d cycles...\n\n", iCycles);
    shrDeltaT(2); 
    for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += SobelFilterGPU (uiInput, uiOutput);
    }

    double dRoundtripTime = shrDeltaT(2)/(double)iCycles;
    dProcessingTime /= (double)iCycles;

    shrLogEx(LOGBOTH | MASTER, 0, "oclSobelFilter, Throughput = %.4f M RGB Pixels/s, Time = %.5f s, Size = %u RGB Pixels, NumDevsUsed = %u, Workgroup = %u\n", 
           (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime, dProcessingTime, (uiImageWidth * uiImageHeight), GpuDevMngr->uiUsefulDevCt, szLocalWorkSize[0] * szLocalWorkSize[1]); 
    shrLog("\nRoundTrip Time = %.5f s, Equivalent FPS = %.1f\n\n", dRoundtripTime, 1.0/dRoundtripTime);

    cl_uint* uiGolden = (cl_uint*)malloc(szBuffBytes);
    SobelFilterHost(uiInput, uiGolden, uiImageWidth, uiImageHeight, fThresh);

    shrLog("Comparing GPU Result to CPU Result...\n"); 
    shrBOOL bMatch = shrCompareuit(uiGolden, uiOutput, (uiImageWidth * uiImageHeight), 1.0f, 0.0001f);
    shrLog("\nGPU Result %s CPU Result within tolerance...\n", (bMatch == shrTRUE) ? "matches" : "DOESN'T match"); 

    free(uiGolden);
    Cleanup((bMatch == shrTRUE) ? EXIT_SUCCESS : EXIT_FAILURE);
}

void ShowMenuItems()
{
    shrLog("  Right Click on Mouse for Menu\n\n"); 
    shrLog("  or\n\nPress:\n\n");
	shrLog("  <spacebar> to toggle Filter On/Off\n");
	shrLog("  <F> key to toggle between FullScreen and Windowed\n");
    shrLog("  <P> key to toggle Processing between GPU and CPU\n");
	shrLog("  <-/+> Change Threshold (-/+ 10.0)\n");
	shrLog("  <ESC> to Quit\n\n"); 
}

void Cleanup(int iExitCode)
{
    shrLog("\nStarting Cleanup...\n\n");

    if(cpProgram)clReleaseProgram(cpProgram);
    for (cl_uint i = 0; i < GpuDevMngr->uiUsefulDevCt; i++)
    {
        if(ckSobel[i])clReleaseKernel(ckSobel[i]);
        if(cmDevBufIn[i])clReleaseMemObject(cmDevBufIn[i]);
        if(cmDevBufOut[i])clReleaseMemObject(cmDevBufOut[i]);
    }
    if(uiInput)clEnqueueUnmapMemObject(cqCommandQueue[0], cmPinnedBufIn, (void*)uiInput, 0, NULL, NULL);
    if(uiOutput)clEnqueueUnmapMemObject(cqCommandQueue[0], cmPinnedBufOut, (void*)uiOutput, 0, NULL, NULL);
    if(cmPinnedBufIn)clReleaseMemObject(cmPinnedBufIn);
    if(cmPinnedBufOut)clReleaseMemObject(cmPinnedBufOut);
    for (cl_uint i = 0; i < GpuDevMngr->uiUsefulDevCt; i++)
    {
        if(cqCommandQueue[i])clReleaseCommandQueue(cqCommandQueue[i]);
    }
    if(cxGPUContext)clReleaseContext(cxGPUContext);

    if(cSourceCL)free(cSourceCL);
    if(cPathAndName)free(cPathAndName);
    if(cmDevBufIn) delete [] cmDevBufIn;
    if(cmDevBufOut) delete [] cmDevBufOut;
    if(szAllocDevBytes) delete [] szAllocDevBytes;
    if(uiInHostPixOffsets) delete [] uiInHostPixOffsets;
    if(uiOutHostPixOffsets) delete [] uiOutHostPixOffsets;
    if(uiDevImageHeight) delete [] uiDevImageHeight;
    if(GpuDevMngr) delete GpuDevMngr;
    if(cqCommandQueue) delete [] cqCommandQueue;

    if (!bQATest)
    {
        DeInitGL();
    }

    shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutableName);

    shrQAFinishExit2(bQATest, *pArgc, (const char **)pArgv, ( iExitCode == EXIT_SUCCESS ) ? QA_PASSED : QA_FAILED);
}
