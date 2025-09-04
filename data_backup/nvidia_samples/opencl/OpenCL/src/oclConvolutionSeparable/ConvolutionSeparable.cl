#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

__kernel __attribute__((reqd_work_group_size(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1)))
void convolutionRows(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch
){
    __local float l_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    const int baseX = (get_group_id(0) * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + get_local_id(0);
    const int baseY = get_group_id(1) * ROWS_BLOCKDIM_Y + get_local_id(1);

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];

    for(int i = 0; i < ROWS_HALO_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X >= 0) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

    for(int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
        l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X]  = (baseX + i * ROWS_BLOCKDIM_X < imageW) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++){
        float sum = 0;

        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += c_Kernel[KERNEL_RADIUS - j] * l_Data[get_local_id(1)][get_local_id(0) + i * ROWS_BLOCKDIM_X + j];

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}

__kernel __attribute__((reqd_work_group_size(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1)))
void convolutionColumns(
    __global float *d_Dst,
    __global float *d_Src,
    __constant float *c_Kernel,
    int imageW,
    int imageH,
    int pitch
){
    __local float l_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    const int baseX = get_group_id(0) * COLUMNS_BLOCKDIM_X + get_local_id(0);
    const int baseY = (get_group_id(1) * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + get_local_id(1);
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];

    for(int i = 0; i < COLUMNS_HALO_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y] = (baseY + i * COLUMNS_BLOCKDIM_Y >= 0) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    for(int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
        l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y]  = (baseY + i * COLUMNS_BLOCKDIM_Y < imageH) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for(int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++){
        float sum = 0;

        for(int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            sum += c_Kernel[KERNEL_RADIUS - j] * l_Data[get_local_id(0)][get_local_id(1) + i * COLUMNS_BLOCKDIM_Y + j];

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}


