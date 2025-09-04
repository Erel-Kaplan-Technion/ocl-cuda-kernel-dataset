#define BLOCK_DIM 16

__kernel void transpose(__global float *odata, __global float *idata, int offset, int width, int height, __local float* block)
{
	// read the matrix tile into shared memory
	unsigned int xIndex = get_global_id(0);
	unsigned int yIndex = get_global_id(1);

	if((xIndex + offset < width) && (yIndex < height))
	{
		unsigned int index_in = yIndex * width + xIndex + offset;
		block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// write the transposed matrix tile to global memory
	xIndex = get_group_id(1) * BLOCK_DIM + get_local_id(0);
	yIndex = get_group_id(0) * BLOCK_DIM + get_local_id(1);
	if((xIndex < height) && (yIndex + offset < width))
    {
		unsigned int index_out = yIndex * height + xIndex;
		odata[index_out] = block[get_local_id(0)*(BLOCK_DIM+1)+get_local_id(1)];
	}
}

__kernel void transpose_naive(__global float *odata, __global float* idata, int offset, int width, int height)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);
    
    if (xIndex + offset < width && yIndex < height)
    {
        unsigned int index_in  = xIndex + offset + width * yIndex;
        unsigned int index_out = yIndex + height * xIndex;
        odata[index_out] = idata[index_in]; 
    }
}


__kernel void simple_copy(__global float *odata, __global float* idata, int offset, int width, int height)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);
    
    if (xIndex + offset < width && yIndex < height)
    {
        unsigned int index_in  = xIndex + offset + width * yIndex;
        odata[index_in] = idata[index_in]; 
    }
}

__kernel void shared_copy(__global float *odata, __global float *idata, int offset, int width, int height, __local float* block)
{
	// read the matrix tile into shared memory
	unsigned int xIndex = get_global_id(0);
	unsigned int yIndex = get_global_id(1);

    unsigned int index_in = yIndex * width + xIndex + offset;
	if((xIndex + offset< width) && (yIndex < height))
	{
		block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)] = idata[index_in];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if((xIndex < height) && (yIndex+ offset < width))
    {
		odata[index_in] = block[get_local_id(1)*(BLOCK_DIM+1)+get_local_id(0)];
	}
}


__kernel void uncoalesced_copy(__global float *odata, __global float* idata, int offset, int width, int height)
{
    unsigned int xIndex = get_global_id(0);
    unsigned int yIndex = get_global_id(1);
    
    if (xIndex + offset < width && yIndex < height)
    {
        unsigned int index_in  = yIndex + height * (xIndex+ offset);
        odata[index_in] = idata[index_in]; 
    }
}
