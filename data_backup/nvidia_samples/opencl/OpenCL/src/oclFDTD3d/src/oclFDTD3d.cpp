#include "oclFDTD3d.h"

#include <oclUtils.h>
#include <shrQATest.h>
#include <iostream>
#include <iomanip>

#include "FDTD3dReference.h"
#include "FDTD3dGPU.h"

const char* clSourceFile = "FDTD3d.cl";
const char *shrLogFile = "oclFDTD3d.txt";

bool runTest(int argc, const char **argv);
void showHelp(const int argc, const char **argv);

int main(int argc, char **argv)
{
    bool bTestResult = true;
    shrQAStart(argc, argv);

    shrSetLogFileName(shrLogFile);
    shrLog("%s Starting...\n\n", argv[0]);

    // Check help flag
    if (shrCheckCmdLineFlag(argc, (const char **)argv, "help")) {
        shrLog("Displaying help on console\n");
        showHelp(argc, (const char **)argv);
    } else {
        // Execute
        bTestResult = runTest(argc, (const char **)argv);
        oclCheckErrorEX(bTestResult, true, NULL);
    }

    // Finish
    shrQAFinishExit( argc, (const char **)argv, (bTestResult ? QA_PASSED : QA_FAILED) );
}

void showHelp(const int argc, const char **argv)
{
    if (argc > 0)
        std::cout << std::endl << argv[0] << std::endl;
    std::cout << std::endl << "Syntax:" << std::endl;
    std::cout << std::left;
    std::cout << "    " << std::setw(20) << "--device=<device>" << "Specify device to use for execution" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimx=<N>" << "Specify number of elements in x direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimy=<N>" << "Specify number of elements in y direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimz=<N>" << "Specify number of elements in z direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--radius=<N>" << "Specify radius of stencil" << std::endl;
    std::cout << "    " << std::setw(20) << "--timesteps=<N>" << "Specify number of timesteps" << std::endl;
    std::cout << "    " << std::setw(20) << "--work-group-size=<N>" << "Specify work group size" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << std::setw(20) << "--noprompt" << "Skip prompt before exit" << std::endl;
    std::cout << std::endl;
}

bool runTest(int argc, const char **argv)
{
    bool ok = true;

    float *host_output;
    float *device_output;
    float *input;
    float *coeff;

    int defaultDim;
    int dimx;
    int dimy;
    int dimz;
    int outerDimx;
    int outerDimy;
    int outerDimz;
    int radius;
    int timesteps;
    size_t volumeSize;
    memsize_t memsize;

    const float lowerBound = 0.0f;
    const float upperBound = 1.0f;

    // Determine default dimensions
    shrLog("Set-up, based upon target device GMEM size...\n");
    if (ok)
    {
        // Get the memory size of the target device
        shrLog(" getTargetDeviceGlobalMemSize\n");
        ok = getTargetDeviceGlobalMemSize(&memsize, argc, argv);
    }
    if (ok)
    {
        memsize /= 2;
        
        defaultDim = (int)floor(pow((memsize / (2.0 * sizeof(float))), 1.0/3.0));

        int roundTarget = 128 / sizeof(float);
        defaultDim = defaultDim / roundTarget * roundTarget;
        defaultDim -= k_radius_default * 2;

        // Check dimension is valid
        if (defaultDim < k_dim_min)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -1000, STDERROR);
            shrLog("\tinsufficient device memory (maximum volume on device is %d, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
            ok = false;
        }

        else if (defaultDim > k_dim_max)
        {
            defaultDim = k_dim_max;
        }
    }

    // For QA testing, override default volume size
    if (ok)
    {
        if (shrCheckCmdLineFlag(argc, argv, "qatest"))
        {
            defaultDim = MIN(defaultDim, k_dim_qa);
        }
    }

    // Parse command line arguments
    if (ok)
    {
        char *dim = 0;
        if (shrGetCmdLineArgumentstr(argc, argv, "dimx", &dim))
        {
            dimx = (int)atoi(dim);
            if (dimx < k_dim_min || dimx > k_dim_max)
            {
                shrLogEx(LOGBOTH | ERRORMSG, -1001, STDERROR);
                shrLog("\tdimx out of range (%d requested, must be between %d and %d), see header files for details.\n", dimx, k_dim_min, k_dim_max);
                ok = false;
            }
        }
        else
        {
            dimx = defaultDim;
        }
        if (shrGetCmdLineArgumentstr(argc, argv, "dimy", &dim))
        {
            dimy = (int)atoi(dim);
            if (dimy < k_dim_min || dimy > k_dim_max)
            {
                shrLogEx(LOGBOTH | ERRORMSG, -1002, STDERROR);
                shrLog("\tdimy out of range (%d requested, must be between %d and %d), see header files for details.\n", dimy, k_dim_min, k_dim_max);
                ok = false;
            }
        }
        else
        {
            dimy = defaultDim;
        }
        if (shrGetCmdLineArgumentstr(argc, argv, "dimz", &dim))
        {
            dimz = (int)atoi(dim);
            if (dimz < k_dim_min || dimz > k_dim_max)
            {
                shrLogEx(LOGBOTH | ERRORMSG, -1003, STDERROR);
                shrLog("\tdimz out of range (%d requested, must be between %d and %d), see header files for details.\n", dimz, k_dim_min, k_dim_max);
                ok = false;
            }
        }
        else
        {
            dimz = defaultDim;
        }
        if (shrGetCmdLineArgumentstr(argc, argv, "radius", &dim))
        {
            radius = (int)atoi(dim);
            if (radius < k_radius_min || radius >= k_radius_max)
            {
                shrLogEx(LOGBOTH | ERRORMSG, -1004, STDERROR);
                shrLog("\tradius out of range (%d requested, must be between %d and %d), see header files for details.\n", radius, k_radius_min, k_radius_max);
                ok = false;
            }
        }
        else
        {
            radius = k_radius_default;
        }
        if (shrGetCmdLineArgumentstr(argc, argv, "timesteps", &dim))
        {
            timesteps = (int)atoi(dim);
            if (timesteps < k_timesteps_min || radius >= k_timesteps_max)
            {
                shrLogEx(LOGBOTH | ERRORMSG, -1005, STDERROR);
                shrLog("\ttimesteps out of range (%d requested, must be between %d and %d), see header files for details.\n", timesteps, k_timesteps_min, k_timesteps_max);
                ok = false;
            }
        }
        else
        {
            timesteps = k_timesteps_default;
        }
        if (dim)
            free(dim);
    }

    // Determine volume size
    if (ok)
    {
        outerDimx = dimx + 2 * radius;
        outerDimy = dimy + 2 * radius;
        outerDimz = dimz + 2 * radius;
        volumeSize = outerDimx * outerDimy * outerDimz;
    }
    
    // Allocate memory
    if (ok)
    {
        shrLog(" calloc host_output\n");
        if ((host_output = (float *)calloc(volumeSize, sizeof(float))) == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -1006, STDERROR);
            shrLog("\tInsufficient memory for host_output calloc, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(" malloc input\n");
        if ((input = (float *)malloc(volumeSize * sizeof(float))) == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -1007, STDERROR);
            shrLog("\tInsufficient memory for input malloc, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }
    if (ok)
    {
        shrLog(" malloc coeff\n");
        if ((coeff = (float *)malloc((radius + 1) * sizeof(float))) == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -1008, STDERROR);
            shrLog("\tInsufficient memory for coeff malloc, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }

    // Create coefficients
    if (ok)
    {
        for (int i = 0 ; i <= radius ; i++)
        {
            coeff[i] = 0.1f;
        }
    }

    // Generate data
    if (ok)
    {
        shrLog(" generateRandomData\n\n");
        generateRandomData(input, outerDimx, outerDimy, outerDimz, lowerBound, upperBound);
    }

    if (ok)
    {
        shrLog("FDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);
    }

    // Execute on the host
    if (ok)
    {
        shrLog("fdtdReference...\n");
        ok = fdtdReference(host_output, input, coeff, dimx, dimy, dimz, radius, timesteps);
        shrLog("fdtdReference complete\n");
    }

    // Allocate memory
    if (ok)
    {
        shrLog(" calloc device_output\n");
        if ((device_output = (float *)calloc(volumeSize, sizeof(float))) == NULL)
        {
            shrLogEx(LOGBOTH | ERRORMSG, -1009, STDERROR);
            shrLog("\tInsufficient memory for device output calloc, please try a smaller volume (use --help for syntax).\n");
            ok = false;
        }
    }

    // Execute on the device
    if (ok)
    {
        shrLog("fdtdGPU...\n");
        ok = fdtdGPU(device_output, input, coeff, dimx, dimy, dimz, radius, timesteps, argc, argv);
        shrLog("fdtdGPU complete\n");
    }

    // Compare the results
    if (ok)
    {
        float tolerance = 0.0001f;
        shrLog("\nCompareData (tolerance %f)...\n", tolerance);
        ok = compareData(device_output, host_output, dimx, dimy, dimz, radius, tolerance);
    }

    return ok;
}
