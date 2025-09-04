float4 rgbaUintToFloat4(unsigned int c)
{
    float4 rgba;
    rgba.x = c & 0xff;
    rgba.y = (c >> 8) & 0xff;
    rgba.z = (c >> 16) & 0xff;
    rgba.w = (c >> 24) & 0xff;
    return rgba;
}

unsigned int rgbaFloat4ToUint(float4 rgba, float fScale)
{
    unsigned int uiPackedPix = 0U;
    uiPackedPix |= 0x000000FF & (unsigned int)(rgba.x * fScale);
    uiPackedPix |= 0x0000FF00 & (((unsigned int)(rgba.y * fScale)) << 8);
    uiPackedPix |= 0x00FF0000 & (((unsigned int)(rgba.z * fScale)) << 16);
    uiPackedPix |= 0xFF000000 & (((unsigned int)(rgba.w * fScale)) << 24);
    return uiPackedPix;
}


#ifdef USETEXTURE
    __kernel void BoxRowsTex( __read_only image2d_t SourceRgbaTex, __global unsigned int* uiDest, sampler_t RowSampler, 
                              unsigned int uiWidth, unsigned int uiHeight, int iRadius, float fScale)
    {
	    size_t globalPosY = get_global_id(0);
        size_t szBaseOffset = mul24(globalPosY, uiWidth);

        if (globalPosY < uiHeight) 
        {
            float4 f4Sum = (float4)0.0f;

            for(int x = -iRadius; x <= iRadius; x++)
            {
                int2 pos = {x , globalPosY};
                f4Sum += convert_float4(read_imageui(SourceRgbaTex, RowSampler, pos));  
            }
            uiDest[szBaseOffset] = rgbaFloat4ToUint(f4Sum, fScale);

            int2 pos = {0, globalPosY};
            for(unsigned int x = 1; x < uiWidth; x++)
            {
                pos.x = x + iRadius;
                f4Sum += convert_float4(read_imageui(SourceRgbaTex, RowSampler, pos));  

                pos.x = x - iRadius - 1;
                f4Sum -= convert_float4(read_imageui(SourceRgbaTex, RowSampler, pos));  

                uiDest[szBaseOffset + x] = rgbaFloat4ToUint(f4Sum, fScale);
            }
        }
    }
#endif


#ifdef USELMEM

    __kernel void BoxRowsLmem( __global const uchar4* uc4Source, __global unsigned int* uiDest,
                               __local uchar4* uc4LocalData,
                               unsigned int uiWidth, unsigned int uiHeight, int iRadius, int iRadiusAligned, 
                               float fScale, unsigned int uiNumOutputPix)
    {
        int globalPosX = ((int)get_group_id(0) * uiNumOutputPix) + (int)get_local_id(0) - iRadiusAligned;
        int globalPosY = (int)get_group_id(1);
        int iGlobalOffset = globalPosY * uiWidth + globalPosX;

        if (globalPosX >= 0 && globalPosX < uiWidth)
        {
            uc4LocalData[get_local_id(0)] = uc4Source[iGlobalOffset];
        }
        else 
        {
            uc4LocalData[get_local_id(0)].xyzw = (uchar4)0; 
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if((globalPosX >= 0) && (globalPosX < uiWidth) && (get_local_id(0) >= iRadiusAligned) && (get_local_id(0) < (iRadiusAligned + (int)uiNumOutputPix)))
        {
            float4 f4Sum = (float4)0.0f;

            int iOffsetX = (int)get_local_id(0) - iRadius;
            int iLimit = iOffsetX + (2 * iRadius) + 1;
            for(iOffsetX; iOffsetX < iLimit; iOffsetX++)
            {
                f4Sum.x += uc4LocalData[iOffsetX].x;
                f4Sum.y += uc4LocalData[iOffsetX].y;
                f4Sum.z += uc4LocalData[iOffsetX].z;
                f4Sum.w += uc4LocalData[iOffsetX].w; 
            }

            uiDest[iGlobalOffset] = rgbaFloat4ToUint(f4Sum, fScale);
        }
    }
#endif


__kernel void BoxColumns(__global unsigned int* uiInputImage, __global unsigned int* uiOutputImage, 
                         unsigned int uiWidth, unsigned int uiHeight, int iRadius, float fScale)
{
	size_t globalPosX = get_global_id(0);
    uiInputImage = &uiInputImage[globalPosX];
    uiOutputImage = &uiOutputImage[globalPosX];

    float4 f4Sum;
    f4Sum = rgbaUintToFloat4(uiInputImage[0]) * (float4)(iRadius);
    for (int y = 0; y < iRadius + 1; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[y * uiWidth]);
    }
    uiOutputImage[0] = rgbaFloat4ToUint(f4Sum, fScale);
    for(int y = 1; y < iRadius + 1; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[0]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
    
    for(int y = iRadius + 1; y < uiHeight - iRadius; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(y + iRadius) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }

    for (int y = uiHeight - iRadius; y < uiHeight; y++) 
    {
        f4Sum += rgbaUintToFloat4(uiInputImage[(uiHeight - 1) * uiWidth]);
        f4Sum -= rgbaUintToFloat4(uiInputImage[((y - iRadius) * uiWidth) - uiWidth]);
        uiOutputImage[y * uiWidth] = rgbaFloat4ToUint(f4Sum, fScale);
    }
}
