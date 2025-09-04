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

#include <memory>
#include <iostream>
#include <cassert>

#include <oclUtils.h>
#include <shrQATest.h>

#ifndef min
#define min(a,b) (a < b ? a : b);
#endif

extern "C" double BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                                unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);

#define REFRESH_DELAY	  10 //ms

cl_uint uiNumOutputPix = 64;
cl_uint iRadius = 10;
float fScale = 1.0f/(2.0f * iRadius + 1.0f);
cl_int iRadiusAligned;

const char* cImageFile = "lenaRGB.ppm";
unsigned int uiImageWidth = 0;
unsigned int uiImageHeight = 0;
unsigned int* uiInput = NULL;
unsigned int* uiTemp = NULL;

int iGLUTWindowHandle;
int iGLUTMenuHandle;
int iGraphicsWinPosX = 0;
int iGraphicsWinPosY = 0;
int iGraphicsWinWidth = 800;
int iGraphicsWinHeight = 800;
int iGraphicsWinWidthNonFS = 800;
int iGraphicsWinHeightNonFS = 800;
int iFrameCount = 0;
int iFrameTrigger = 90;
int iFramesPerSec = 60;
double dProcessingTime = 0.0;
bool bGLinteropSupported = false;
GLint iVsyncState;

const char* cProcessor [] = {"OpenCL GPU", "Host C++ CPU"};
bool bFilter = true;
bool bFullScreen = false;
bool bGLinterop = false;
int iProcFlag = 0;
shrBOOL bNoPrompt = shrFALSE;
shrBOOL bQATest = shrFALSE;
shrBOOL bUseLmem = shrFALSE;
int iTestSets = 3;

const char* clSourcefile = "BoxFilter.cl";
char* cPathAndName = NULL;
char* cSourceCL = NULL;
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_command_queue cqCommandQueue;
cl_device_id* cdDevices = NULL;
cl_uint uiNumDevsUsed = 1;
cl_program cpProgram;
cl_kernel ckBoxRowsLmem;
cl_kernel ckBoxRowsTex;
cl_kernel ckBoxColumns;
cl_mem cmDevBufIn;
cl_mem cmDevBufTemp;
cl_mem cmDevBufOut;
cl_mem cmCL_PBO=0;
cl_image_format InputFormat;
cl_sampler RowSampler;
size_t szBuffBytes;
size_t szGlobalWorkSize[2];
size_t szLocalWorkSize[2];
size_t szMaxWorkgroupSize = 512;
size_t szParmDataBytes;
size_t szKernelLength;
cl_int ciErrNum;

GLuint tex_screen;
GLuint pbo;

const char* cpExecutableName;

int pArgc = 0;
char **pArgv = NULL;

double BoxFilterGPU(unsigned int* uiInputImage, cl_mem cmOutputBuffer, 
                    unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);
void ResetKernelArgs(unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);

void InitGlut(int* argc, char** argv);
void InitGlew();
void DeInitGL();
void DisplayGL();
void Reshape(int w, int h);
void Idle(void);
void KeyboardGL(unsigned char key, int x, int y);
void MenuGL(int i);
void timerEvent(int value);

void createPBO(GLuint* pbo, int image_width, int image_height);
void deletePBO(GLuint* pbo);
void createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y);
void deleteTexture(GLuint* tex);
void displayTexture(GLuint tex);

void TestNoGL();
void TriggerFPSUpdate();
static inline size_t DivUp(size_t dividend, size_t divisor);
void ShowMenuItems();
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;

int main(int argc, char** argv)
{
	pArgc = argc;
	pArgv = argv;

	shrQAStart(argc, argv);
	cpExecutableName = argv[0];
    shrSetLogFileName ("oclBoxFilter.txt");
    shrLog("%s Starting, using %s...\n\n", argv[0], clSourcefile); 

    bNoPrompt = shrCheckCmdLineFlag(argc, (const char**)argv, "noprompt");
    bQATest = shrCheckCmdLineFlag(argc, (const char**)argv, "qatest");
    bUseLmem = shrCheckCmdLineFlag(argc, (const char**)argv, "lmem");
    bGLinterop = (shrCheckCmdLineFlag(argc, (const char**)argv, "GLinterop") == shrTRUE);

    if (!(bQATest))
    {
        ShowMenuItems();
    }

    cPathAndName = shrFindFilePath(cImageFile, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    ciErrNum = shrLoadPPM4ub(cPathAndName, (unsigned char **)&uiInput, &uiImageWidth, &uiImageHeight);
    oclCheckErrorEX(ciErrNum, shrTRUE, pCleanup);
    shrLog("Image Width = %i, Height = %i, bpp = %i, Mask Radius = %i\n", uiImageWidth, uiImageHeight, sizeof(unsigned int)<<3, iRadius);
    shrLog("Using %s for Row Processing\n\n", bUseLmem ? "Local Memory (lmem)" : "2d Image (Texture)");
    
    szBuffBytes = uiImageWidth * uiImageHeight * sizeof (unsigned int);
    uiTemp = (unsigned int*)malloc(szBuffBytes);
    shrLog("Allocate Host Image Buffers...\n"); 

    shrLog("%sInitGlut, InitGlew...\n", bQATest ? "Skipping " : "Calling "); 
    if (!(bQATest))
    {
        InitGlut(&argc, argv);
        InitGlew();
    }

    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clGetPlatformID...\n"); 

    cl_uint uiNumDevices = 0;
    cl_uint uiTargetDevice = 0;
    cl_uint uiNumComputeUnits;
    shrLog("Get the Device info and select Device...\n");
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiNumDevices);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cdDevices = (cl_device_id *)malloc(uiNumDevices * sizeof(cl_device_id) );
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiNumDevices, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Set target device and Query number of compute units on uiTargetDevice
    shrLog("  # of Devices Available = %u\n", uiNumDevices); 
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiTargetDevice)== shrTRUE) 
    {
        uiTargetDevice = CLAMP(uiTargetDevice, 0, (uiNumDevices - 1));
    }
    shrLog("  Using Device %u: ", uiTargetDevice); 
    oclPrintDevName(LOGBOTH, cdDevices[uiTargetDevice]);
    ciErrNum = clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uiNumComputeUnits), &uiNumComputeUnits, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("\n  # of Compute Units = %u\n", uiNumComputeUnits); 

    // Check for GL interop capability (if using GL)
    if(!bQATest)
    {
        char extensions[1024];
        ciErrNum = clGetDeviceInfo(cdDevices[uiTargetDevice], CL_DEVICE_EXTENSIONS, 1024, extensions, 0);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        
        #if defined (__APPLE__) || defined(MACOSX)
            bGLinteropSupported = std::string(extensions).find("cl_APPLE_gl_sharing") != std::string::npos;
        #else
            bGLinteropSupported = std::string(extensions).find("cl_khr_gl_sharing") != std::string::npos;
        #endif
    }

    //Create the context
    if(bGLinteropSupported) 
    {
        // Define OS-specific context properties and create the OpenCL context
        #if defined (__APPLE__)
            CGLContextObj kCGLContext = CGLGetCurrentContext();
            CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
            cl_context_properties props[] = 
            {
                CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup, 
                0 
            };
            cxGPUContext = clCreateContext(props, 0,0, NULL, NULL, &ciErrNum);
        #else
            #ifdef UNIX
                cl_context_properties props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
                    CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                    0
                };
                cxGPUContext = clCreateContext(props, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
            #else // Win32
                cl_context_properties props[] = 
                {
                    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
                    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
                    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
                    0
                };
                cxGPUContext = clCreateContext(props, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
            #endif
        #endif
        shrLog("clCreateContext, GL Interop supported...\n"); 
    } 
    else 
    {
        bGLinterop = false;
        cxGPUContext = clCreateContext(0, uiNumDevsUsed, &cdDevices[uiTargetDevice], NULL, NULL, &ciErrNum);
        shrLog("clCreateContext, GL Interop %s...\n", bQATest ? "N/A" : "not supported"); 
    }
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    // Create a command-queue 
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiTargetDevice], 0, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateCommandQueue...\n"); 

    // Allocate OpenCL object for the source data
    if (bUseLmem)
    {
        // Buffer in device GMEM
        cmDevBufIn = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY, szBuffBytes, NULL, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clCreateBuffer (Input buffer, device GMEM)...\n");
    }
    else 
    {
        // 2D Image (Texture) on device
        InputFormat.image_channel_order = CL_RGBA;
        InputFormat.image_channel_data_type = CL_UNSIGNED_INT8;
        cmDevBufIn = clCreateImage2D(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &InputFormat, 
                                     uiImageWidth, uiImageHeight, 
                                     uiImageWidth * sizeof(unsigned int), uiInput, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clCreateImage2D (Input buffer, device GMEM)...\n");

        RowSampler = clCreateSampler(cxGPUContext, false, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clCreateSampler (Non-Normalized Coords, CL_ADDRESS_CLAMP, CL_FILTER_NEAREST)...\n");
    }

    // Allocate the OpenCL intermediate and result buffer memory objects on the device GMEM
    cmDevBufTemp = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    cmDevBufOut = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, szBuffBytes, NULL, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateBuffer (Intermediate and Output buffers, device GMEM)...\n"); 

    // Create OpenCL representation of OpenGL PBO
    if(bGLinteropSupported)
    {
        cmCL_PBO = clCreateFromGLBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, pbo, &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clCreateFromGLBuffer (cmCL_PBO)...\n"); 
    }

    // Read the OpenCL kernel source in from file
    free(cPathAndName);
    cPathAndName = shrFindFilePath(clSourcefile, argv[0]);
    oclCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "// My comment\n", &szKernelLength);
    oclCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);
    shrLog("oclLoadProgSource...\n"); 

    // Create the program 
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&cSourceCL, &szKernelLength, &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateProgramWithSource...\n"); 

    std::string sBuildOpts = " -cl-fast-relaxed-math"; 
    sBuildOpts  += bUseLmem ? " -D USELMEM" : " -D USETEXTURE";

    // mac
    #ifdef MAC
        sBuildOpts  += " -DMAC";
    #endif

    ciErrNum = clBuildProgram(cpProgram, 0, NULL, sBuildOpts.c_str(), NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, cdDevices[uiTargetDevice]);
        oclLogPtx(cpProgram, cdDevices[uiTargetDevice], "oclBoxFilter.ptx");
        shrQAFinish(argc, (const char **)argv, QA_FAILED);
        Cleanup(EXIT_FAILURE);
    }
    shrLog("clBuildProgram...\n"); 

    if (bUseLmem)
    {
        ckBoxRowsLmem = clCreateKernel(cpProgram, "BoxRowsLmem", &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clCreateKernel (BoxRowsLmem)...\n"); 

        ciErrNum = clGetKernelWorkGroupInfo(ckBoxRowsLmem, cdDevices[uiTargetDevice], 
                                            CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &szMaxWorkgroupSize, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    }
    else
    {
        ckBoxRowsTex = clCreateKernel(cpProgram, "BoxRowsTex", &ciErrNum);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clCreateKernel (BoxRowsTex)...\n"); 
    }
    ckBoxColumns = clCreateKernel(cpProgram, "BoxColumns", &ciErrNum);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clCreateKernel (BoxColumns)...\n"); 

    // set the kernel args
    ResetKernelArgs(uiImageWidth, uiImageHeight, iRadius, fScale);

    // init running timers
    shrDeltaT(0);   // timer 0 used for computation timing 
    shrDeltaT(1);   // timer 1 used for fps computation

    if (!(bQATest))
    {
        glutMainLoop();
    }
    else 
    {
        TestNoGL();
    }

    shrQAFinish2(bQATest, argc, (const char **)argv, QA_PASSED);
    Cleanup(EXIT_SUCCESS);
}

void ResetKernelArgs(unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    if (bUseLmem)
    {
        // (lmem version)
        iRadiusAligned = ((r + 15)/16) * 16;
        if (szMaxWorkgroupSize < (iRadiusAligned + uiNumOutputPix + r))
        {
            uiNumOutputPix = (cl_uint)szMaxWorkgroupSize - iRadiusAligned - r;   
        }
        ciErrNum = clSetKernelArg(ckBoxRowsLmem, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 2, (iRadiusAligned + uiNumOutputPix + r) * sizeof(cl_uchar4), NULL);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 3, sizeof(unsigned int), (void*)&uiWidth);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 4, sizeof(unsigned int), (void*)&uiHeight);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 5, sizeof(int), (void*)&r);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 6, sizeof(int), (void*)&iRadiusAligned);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 7, sizeof(float), (void*)&fScale);
        ciErrNum |= clSetKernelArg(ckBoxRowsLmem, 8, sizeof(unsigned int), (void*)&uiNumOutputPix);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clSetKernelArg (0-8) ckBoxRowsLmem...\n"); 
    }
    else
    {
        // (Image/texture version)
        ciErrNum = clSetKernelArg(ckBoxRowsTex, 0, sizeof(cl_mem), (void*)&cmDevBufIn);
        ciErrNum |= clSetKernelArg(ckBoxRowsTex, 1, sizeof(cl_mem), (void*)&cmDevBufTemp);
        ciErrNum |= clSetKernelArg(ckBoxRowsTex, 2, sizeof(cl_sampler), &RowSampler); 
        ciErrNum |= clSetKernelArg(ckBoxRowsTex, 3, sizeof(unsigned int), (void*)&uiWidth);
        ciErrNum |= clSetKernelArg(ckBoxRowsTex, 4, sizeof(unsigned int), (void*)&uiHeight);
        ciErrNum |= clSetKernelArg(ckBoxRowsTex, 5, sizeof(int), (void*)&r);
        ciErrNum |= clSetKernelArg(ckBoxRowsTex, 6, sizeof(float), (void*)&fScale);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
        shrLog("clSetKernelArg (0-6) ckBoxRowsTex...\n"); 
    }

    // Set the Argument values for the column kernel
    ciErrNum  = clSetKernelArg(ckBoxColumns, 0, sizeof(cl_mem), (void*)&cmDevBufTemp);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 1, sizeof(cl_mem), (void*)&cmDevBufOut);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 2, sizeof(unsigned int), (void*)&uiWidth);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 3, sizeof(unsigned int), (void*)&uiHeight);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 4, sizeof(int), (void*)&r);
    ciErrNum |= clSetKernelArg(ckBoxColumns, 5, sizeof(float), (void*)&fScale);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    shrLog("clSetKernelArg (0-5) ckBoxColumns...\n\n"); 
}

double BoxFilterGPU(unsigned int* uiInputImage, cl_mem cmOutputBuffer, 
                    unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    ciErrNum = clSetKernelArg(ckBoxColumns, 1, sizeof(cl_mem), (void*)&cmOutputBuffer);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    if (bUseLmem)
    {
        // lmem version
        ciErrNum = clEnqueueWriteBuffer(cqCommandQueue, cmDevBufIn, CL_TRUE, 0, szBuffBytes, uiInputImage, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        szLocalWorkSize[0] = (size_t)(iRadiusAligned + uiNumOutputPix + r);   // Workgroup padded left and right
        szLocalWorkSize[1] = 1;
        szGlobalWorkSize[0] = szLocalWorkSize[0] * DivUp((size_t)uiWidth, (size_t)uiNumOutputPix);
        szGlobalWorkSize[1] = uiHeight;
    }
    else
    {
        // 2D Image (Texture)
        const size_t szTexOrigin[3] = {0, 0, 0};                // Offset of input texture origin relative to host image
        const size_t szTexRegion[3] = {uiWidth, uiHeight, 1};   // Size of texture region to operate on
        ciErrNum = clEnqueueWriteImage(cqCommandQueue, cmDevBufIn, CL_TRUE, 
                                       szTexOrigin, szTexRegion, 0, 0, uiInputImage, 0, NULL, NULL);
        oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

        szLocalWorkSize[0] = uiNumOutputPix;
        szLocalWorkSize[1] = 1;
        szGlobalWorkSize[0]= szLocalWorkSize[0] * DivUp((size_t)uiHeight, szLocalWorkSize[0]);
        szGlobalWorkSize[1] = 1;
    }

    clFinish(cqCommandQueue);
    shrDeltaT(0);

    if (bUseLmem)
    {
        // lmem Version
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBoxRowsLmem, 2, NULL, 
                                          szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
    }
    else 
    {
        // 2D Image (Texture)
        ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBoxRowsTex, 2, NULL, 
                                          szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
    }
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    szLocalWorkSize[0] = 64;
    szLocalWorkSize[1] = 1;
    szGlobalWorkSize[0] = szLocalWorkSize[0] * DivUp((size_t)uiWidth, szLocalWorkSize[0]);
    szGlobalWorkSize[1] = 1;

    ciErrNum = clEnqueueNDRangeKernel(cqCommandQueue, ckBoxColumns, 2, NULL, 
                                      szGlobalWorkSize, szLocalWorkSize, 0, NULL, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    clFinish(cqCommandQueue);
    return shrDeltaT(0);
}

void InitGlut(int* argc, char **argv)
{
    shrLog("  glutInit...\n"); 
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - iGraphicsWinWidth/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - iGraphicsWinHeight/2);
    glutInitWindowSize(iGraphicsWinWidth, iGraphicsWinHeight);
    iGLUTWindowHandle = glutCreateWindow("OpenCL GPU BoxFilter Demo");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif

    // register glut callbacks
    glutKeyboardFunc(KeyboardGL);
    glutDisplayFunc(DisplayGL);
    glutReshapeFunc(Reshape);
    glutIdleFunc(Idle);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);

    iGLUTMenuHandle = glutCreateMenu(MenuGL);
    glutAddMenuEntry("Toggle Filter On/Off <spacebar>", ' ');
    glutAddMenuEntry("Increase Filter Radius [+]", '+');
    glutAddMenuEntry("Decrease Filter Radius [-]", '-');
    glutAddMenuEntry("Toggle Processing between GPU and CPU [p]", 'p');
    glutAddMenuEntry("Toggle OpenGL interop [g]", 'g');
    glutAddMenuEntry("Toggle between Full Screen and Windowed [f]", 'f');
    glutAddMenuEntry("Quit <esc>", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}


void InitGlew()
{
    // Create GL interop buffers
    shrLog("  glewInit...\n"); 
    glewInit();
    shrLog("  createPBO...\n"); 
    createPBO(&pbo, uiImageWidth, uiImageHeight);
    shrLog("  createTexture...\n"); 
    createTexture(&tex_screen, uiImageWidth, uiImageHeight);

    // Disable vertical sync, if supported
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
    // Restore startup Vsync state, if supported
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

    // Delete GL objects
    if(pbo)deletePBO(&pbo);
    if(tex_screen)deleteTexture(&tex_screen);
}

void DisplayGL()
{        
    if (glutGetWindow() == 0)
    {
        shrQAFinish2(false, pArgc, (const char **)pArgv, QA_PASSED);
        Cleanup(EXIT_SUCCESS);
    }

    if (bFilter)
    {
        if (iProcFlag == 0)
        {
            cl_mem cmOutput;
            if(bGLinterop) 
            {
                glFinish();
                clEnqueueAcquireGLObjects(cqCommandQueue, 1, &cmCL_PBO, 0, 0, 0);
                cmOutput = cmCL_PBO;
            } 
            else 
            {
                cmOutput = cmDevBufOut;
            }

            dProcessingTime += BoxFilterGPU (uiInput, cmOutput, uiImageWidth, uiImageHeight, iRadius, fScale);

            if(bGLinterop) 
            {
                clEnqueueReleaseGLObjects(cqCommandQueue, 1, &cmCL_PBO, 0, 0, 0);
                clFinish(cqCommandQueue);
            } 
            else 
            {
                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);    
                void* uiOutput = glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);

                ciErrNum = clEnqueueReadBuffer(cqCommandQueue, cmDevBufOut, CL_TRUE, 0, szBuffBytes, uiOutput, 0, NULL, NULL);
                oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

                glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
                glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
            }
        }
        else 
        {
            glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);    
            unsigned int* uiOutput = (unsigned int*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);

            dProcessingTime += BoxFilterHost (uiInput, uiTemp, uiOutput, uiImageWidth, uiImageHeight, iRadius, fScale);

            glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); 
            glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        }
        
        glBindTexture(GL_TEXTURE_2D, tex_screen);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);    
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, uiImageWidth, uiImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);        
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    else 
    {
        glBindTexture(GL_TEXTURE_2D, tex_screen);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, uiImageWidth, uiImageHeight, GL_RGBA, GL_UNSIGNED_BYTE, uiInput);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    displayTexture(tex_screen);

    glutSwapBuffers();

    if (iFrameCount++ > iFrameTrigger)
    {
        char cTitle[512];

        iFramesPerSec = (int)((double)iFrameCount / shrDeltaT(1));
        dProcessingTime /= (double)iFrameCount; 

#ifdef GPU_PROFILING
        if (bFilter)
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 512, "%s BoxFilter ON | %s | GL Interop %s | W %u , H %u | r = %i | %i fps | Proc. t = %.5f s | %.1f Mpix/s", 
                cProcessor[iProcFlag], bUseLmem ? "LMEM" : "Texture", bGLinterop ? "ON" : "OFF", uiImageWidth, uiImageHeight, iRadius, 
                      iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
            #else
             sprintf(cTitle, "%s BoxFilter ON| %s | GL Interop %s | W %u , H %u |  r = %i | %i fps | Proc. t = %.5f s | %.1f Mpix/s", 
                     cProcessor[iProcFlag], bUseLmem ? "LMEM" : "Texture", bGLinterop ? "ON" : "OFF", uiImageWidth, uiImageHeight, iRadius,  
                     iFramesPerSec, dProcessingTime, (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime);  
            #endif
        }
        else 
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 512, "%s BoxFilter OFF | W %u , H %u | %i fps", 
                      cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iFramesPerSec);  
            #else 
            sprintf(cTitle, "%s BoxFilter OFF | W %u , H %u | %i fps", 
                    cProcessor[iProcFlag], uiImageWidth, uiImageHeight, iFramesPerSec);  
            #endif
        }
#else
        if (bFilter)
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s BoxFilter ON | %s | GL Interop %s | W %u , H %u | r = %i", 
                      cProcessor[iProcFlag], bUseLmem ? "LMEM" : "Texture", bGLinterop ? "ON" : "OFF", uiImageWidth, uiImageHeight, iRadius);  
            #else
             sprintf(cTitle, "%s BoxFilter ON | %s | GL Interop %s | W %u , H %u | r = %i", 
                     cProcessor[iProcFlag], bUseLmem ? "LMEM" : "Texture", bGLinterop ? "ON" : "OFF", uiImageWidth, uiImageHeight, iRadius);  
            #endif
        }
        else 
        {
            #ifdef _WIN32
            sprintf_s(cTitle, 256, "%s BoxFilter OFF | W %u , H %u", 
                    cProcessor[iProcFlag], uiImageWidth, uiImageHeight);  
            #else 
            sprintf(cTitle, "%s BoxFilter OFF | W %u , H %u", 
                    cProcessor[iProcFlag], uiImageWidth, uiImageHeight);  
            #endif
        }
#endif
        glutSetWindowTitle(cTitle);

        shrLog("%s\n", cTitle); 

        if ((bNoPrompt) && (!--iTestSets))
        {
            shrQAFinish2(false, pArgc, (const char **)pArgv, QA_PASSED);
            Cleanup(EXIT_SUCCESS);
        }

        iFrameCount = 0; 
        dProcessingTime = 0.0;
        iFrameTrigger = (iFramesPerSec > 1) ? iFramesPerSec * 2 : 1;
    }
}

void Reshape(int w, int h)
{
    iGraphicsWinHeight = h;
    iGraphicsWinWidth = w;
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
            shrLog("\n%s Processing...\n\n", cProcessor[iProcFlag]);
            break;
        case '+':
        case '=':
            if ((szMaxWorkgroupSize - (((iRadius + 1 + 15)/16) * 16) - iRadius - 1) > 0)iRadius++;
            break;
        case '-':
        case '_':
            if (iRadius > 1)iRadius--;
            break;
        case 'g':
        case 'G':
            if(bGLinteropSupported) 
            {
                bGLinterop = !bGLinterop;
                shrLog("\nGL Interop Toggled %s...\n", bGLinterop ? "ON" : "OFF");
            } 
            else
            {
                shrLog("\nGL Interop not supported\n");
            }
            break;
        case 'F':
        case 'f':
            bFullScreen = !bFullScreen;
            if (bFullScreen)
            {
                iGraphicsWinPosX = glutGet(GLUT_WINDOW_X) - 8;
                iGraphicsWinPosY = glutGet(GLUT_WINDOW_Y) - 30;
                iGraphicsWinWidthNonFS  = min(glutGet(GLUT_WINDOW_WIDTH) , glutGet(GLUT_SCREEN_WIDTH) - 2*iGraphicsWinPosX ); 
                iGraphicsWinHeightNonFS = min(glutGet(GLUT_WINDOW_HEIGHT), glutGet(GLUT_SCREEN_HEIGHT)- 2*iGraphicsWinPosY ); 
                printf("(x,y)=(%d,%d), (w,h)=(%d,%d)\n", iGraphicsWinPosX, iGraphicsWinPosY, iGraphicsWinWidthNonFS, iGraphicsWinHeightNonFS);
                glutFullScreen();
            }
            else
            {
                glutPositionWindow(iGraphicsWinPosX, iGraphicsWinPosY);
                glutReshapeWindow(iGraphicsWinWidthNonFS, iGraphicsWinHeightNonFS);
            }
            shrLog("\nMain Graphics %s...\n", bFullScreen ? "FullScreen" : "Windowed");
            break;
        case ' ':
            bFilter = !bFilter;
            shrLog("\nBoxFilter Toggled %s...\n", bFilter ? "ON" : "OFF");
            break;
        case '\033':  
        case '\015':
        case 'Q':
        case 'q':
			bNoPrompt = shrTRUE;
            shrQAFinish2(false, pArgc, (const char **)pArgv, QA_PASSED);
            Cleanup(EXIT_SUCCESS);
            break;
    }

    fScale = 1.0f/(2 * iRadius + 1);
    ResetKernelArgs(uiImageWidth, uiImageHeight, iRadius, fScale);

    TriggerFPSUpdate();
}

void MenuGL(int i)
{
    KeyboardGL((unsigned char) i, 0, 0);
}

void Idle(void)
{
}

static inline size_t DivUp(size_t dividend, size_t divisor)
{
    return (dividend % divisor == 0) ? (dividend / divisor) : (dividend / divisor + 1);
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
    BoxFilterGPU (uiInput, cmDevBufOut, uiImageWidth, uiImageHeight, iRadius, fScale);
    clFinish(cqCommandQueue);

    const int iCycles = 150;
    dProcessingTime = 0.0;
    shrLog("\nRunning BoxFilterGPU for %d cycles...\n\n", iCycles);
    shrDeltaT(2);
    for (int i = 0; i < iCycles; i++)
    {
        dProcessingTime += BoxFilterGPU (uiInput, cmDevBufOut, uiImageWidth, uiImageHeight, iRadius, fScale);
    }
    clFinish(cqCommandQueue);

    double dRoundtripTime = shrDeltaT(2)/(double)iCycles;
    dProcessingTime /= (double)iCycles;

    shrLogEx(LOGBOTH | MASTER, 0, "oclBoxFilter-%s, Throughput = %.4f M RGBA Pixels/s, Time = %.5f s, Size = %u RGBA Pixels, NumDevsUsed = %u, Workgroup = %u\n", 
                                  bUseLmem ? "lmem" : "texture",
                                  (1.0e-6 * uiImageWidth * uiImageHeight)/dProcessingTime, dProcessingTime, 
                                  (uiImageWidth * uiImageHeight), uiNumDevsUsed, szLocalWorkSize[0] * szLocalWorkSize[1]); 
    shrLog("\nRoundTrip Time = %.5f s, Equivalent FPS = %.1f\n", dRoundtripTime, 1.0/dRoundtripTime);

    shrQAFinish2(true, pArgc, (const char **)pArgv, QA_PASSED);
    Cleanup(EXIT_SUCCESS);
}

void ShowMenuItems()
{
    shrLog("  Right-Click on Mouse for Menu\n\n"); 
    shrLog("  or\n\n  Press:\n\n   <spacebar> to toggle Filter On/Off\n\n");
    shrLog("   \'F\' key to toggle between FullScreen and Windowed\n\n");
    shrLog("   \'+\' key to Increase filter radius\n\n");
    shrLog("   \'-\' key to Decrease filter radius\n\n"); 
    shrLog("   \'G\' key to toggle between OpenGL interop and no-OpenCL interop\n\n");
    shrLog("   \'P\' key to toggle Processing between GPU and CPU\n\n   <esc> to Quit\n\n\n"); 
}

void Cleanup(int iExitCode)
{
    shrLog("\nStarting Cleanup...\n\n");
    if(cSourceCL)free(cSourceCL);
    if(cPathAndName)free(cPathAndName);
    if(uiInput)free(uiInput);
    if(uiTemp)free(uiTemp);
    if(ckBoxColumns)clReleaseKernel(ckBoxColumns);
    if(ckBoxRowsTex)clReleaseKernel(ckBoxRowsTex);
    if(ckBoxRowsLmem)clReleaseKernel(ckBoxRowsLmem);
    if(cpProgram)clReleaseProgram(cpProgram);
    if(RowSampler)clReleaseSampler(RowSampler);
    if(cmDevBufIn)clReleaseMemObject(cmDevBufIn);
    if(cmDevBufTemp)clReleaseMemObject(cmDevBufTemp);
    if(cmDevBufOut)clReleaseMemObject(cmDevBufOut);
    if(cmCL_PBO)clReleaseMemObject(cmCL_PBO);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cdDevices)free(cdDevices);

    if (!bQATest)
    {
        DeInitGL();
    }

    if ((bNoPrompt)||(bQATest))
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cpExecutableName);
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cpExecutableName);
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}

void displayTexture(GLuint texture)
{
    
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
    glBindTexture(GL_TEXTURE_2D, texture);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, iGraphicsWinWidth, iGraphicsWinHeight);

    glBegin(GL_QUADS);

    glTexCoord2f(0.0, 0.0);
    glVertex3f(-1.0, -1.0, 0.5);

    glTexCoord2f(1.0, 0.0);
    glVertex3f(1.0, -1.0, 0.5);

    glTexCoord2f(1.0, 1.0);
    glVertex3f(1.0, 1.0, 0.5);

    glTexCoord2f(0.0, 1.0);
    glVertex3f(-1.0, 1.0, 0.5);

    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void createPBO(GLuint* pbo, int image_width, int image_height)
{
    // set up data parameter
    int num_texels = image_width * image_height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // create buffer object
    shrLog("    glGenBuffers (pbo)...\n"); 
    glGenBuffers(1, pbo);
    shrLog("    glBindBuffer (pbo)...\n"); 
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);

    // buffer data
    shrLog("    glBufferData...\n"); 
    glBufferData(GL_ARRAY_BUFFER, size_tex_data, NULL, GL_DYNAMIC_DRAW);
    shrLog("    glBindBuffer...\n"); 
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void deletePBO(GLuint* pbo)
{
    glBindBuffer(GL_ARRAY_BUFFER, *pbo);
    glDeleteBuffers(1, pbo);
    *pbo = 0;
}

void deleteTexture(GLuint* tex)
{
    glDeleteTextures(1, tex);
    *tex = 0;
}

void createTexture(GLuint* tex_name, unsigned int size_x, unsigned int size_y)
{
    shrLog("    glGenTextures...\n"); 
    glGenTextures(1, tex_name);
    shrLog("    glGenTextures...\n"); 
    glBindTexture(GL_TEXTURE_2D, *tex_name);

    shrLog("    glTexParameteri...\n"); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    shrLog("    glTexImage2D...\n"); 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, size_x, size_y, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}
