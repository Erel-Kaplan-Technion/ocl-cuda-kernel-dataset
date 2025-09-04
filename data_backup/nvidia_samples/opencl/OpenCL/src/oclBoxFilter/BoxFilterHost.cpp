#include <shrUtils.h>

extern "C" double BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                                unsigned int uiWidth, unsigned int uiHeight, int r, float fScale);

unsigned int rgbaFloat4ToUint(const float* rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba[0] * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba[1] * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba[2] * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba[3] * fScale)) << 24);
    return uiPackedPix;
}

void BoxFilterHostX(unsigned char* uiInputImage, unsigned int* uiOutputImage, 
                    unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    for (unsigned int y = 0; y < uiHeight; y++) 
    {
        float f4Sum [4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int x = 0; x <= r; x++) 
        {
            int iBase = (y * uiWidth + x) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];
        }
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);

        for(int x = 1; x <= r; x++) 
        {
            int iBase = (y * uiWidth + x + r) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale);
        }
        
        for(unsigned int x = r + 1; x < uiWidth - r; x++) 
        {
            int iBase = (y * uiWidth + x + r) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];

            iBase = (y * uiWidth + x - r - 1) << 2;
            f4Sum[0] -= uiInputImage[iBase];
            f4Sum[1] -= uiInputImage[iBase + 1];
            f4Sum[2] -= uiInputImage[iBase + 2];
            f4Sum[3] -= uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale);
        }

        for (unsigned int x = uiWidth - r; x < uiWidth; x++) 
        {
            int iBase = (y * uiWidth + x - r - 1) << 2;
            f4Sum[0] -= uiInputImage[iBase];
            f4Sum[1] -= uiInputImage[iBase + 1];
            f4Sum[2] -= uiInputImage[iBase + 2];
            f4Sum[3] -= uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale);
        }
    }
}

void BoxFilterHostY(unsigned char* uiInputImage, unsigned int* uiOutputImage, 
                    unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    for (unsigned int x = 0; x < uiWidth; x++) 
    {
        float f4Sum [4] = {0.0f, 0.0f, 0.0f, 0.0f};

        for (int y = 0; y <= r; y++) 
        {
            int iBase = (y * uiWidth + x) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];
        }
        uiOutputImage[x] = rgbaFloat4ToUint(f4Sum, fScale);

        for(int y = 1; y <= r; y++) 
        {
            int iBase = ((y + r) * uiWidth + x) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale);
        }
        
        for(unsigned int y = r + 1; y < uiHeight - r; y++) 
        {
            int iBase = ((y + r) * uiWidth + x) << 2;
            f4Sum[0] += uiInputImage[iBase];
            f4Sum[1] += uiInputImage[iBase + 1];
            f4Sum[2] += uiInputImage[iBase + 2];
            f4Sum[3] += uiInputImage[iBase + 3];

            iBase = ((y - r) * uiWidth + x - uiWidth) << 2;
            f4Sum[0] -= uiInputImage[iBase];
            f4Sum[1] -= uiInputImage[iBase + 1];
            f4Sum[2] -= uiInputImage[iBase + 2];
            f4Sum[3] -= uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale); 
        }

        for (unsigned int y = uiHeight - r; y < uiHeight; y++) 
        {
            int iBase = ((y - r) * uiWidth + x - uiWidth) << 2;
            f4Sum[0] -= uiInputImage[iBase];
            f4Sum[1] -= uiInputImage[iBase + 1];
            f4Sum[2] -= uiInputImage[iBase + 2];
            f4Sum[3] -= uiInputImage[iBase + 3];

            uiOutputImage[y * uiWidth + x] = rgbaFloat4ToUint(f4Sum, fScale); 
        }
    }
}

double BoxFilterHost(unsigned int* uiInputImage, unsigned int* uiTempImage, unsigned int* uiOutputImage, 
                     unsigned int uiWidth, unsigned int uiHeight, int r, float fScale)
{
    shrDeltaT(0);

    BoxFilterHostX((unsigned char*)uiInputImage, uiTempImage, uiWidth, uiHeight, r, fScale);
    BoxFilterHostY((unsigned char*)uiTempImage, uiOutputImage, uiWidth, uiHeight, r, fScale);

    return shrDeltaT(0);
}
