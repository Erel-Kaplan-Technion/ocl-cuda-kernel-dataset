__kernel void render(__read_only image3d_t volume, sampler_t volumeSampler,  __global uint *d_output, uint imageW, uint imageH, float w)
{
	uint x = get_global_id(0);
    uint y = get_global_id(1);

    // compute normalized texture coordinates
    float u = x / (float) imageW;
    float v = y / (float) imageH;

    // read from 3D texture
    float4 voxel = read_imagef(volume, volumeSampler, (float4)(u,v,w,1.0f));

    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = (y * imageW) + x;
        d_output[i] = voxel.x*255;
    }
}
