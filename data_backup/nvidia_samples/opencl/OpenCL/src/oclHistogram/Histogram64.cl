#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define HISTOGRAM64_BIN_COUNT 64

typedef uint4 data_t;

#define UMUL(a, b)    ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

inline void addByte(__local uchar *l_WorkitemBase, uint data){
    l_WorkitemBase[UMUL(data, HISTOGRAM64_WORKGROUP_SIZE)]++;
}

inline void addWord(__local uchar *l_WorkitemBase, uint data){
    addByte(l_WorkitemBase, (data >>  2) & 0x3FU);
    addByte(l_WorkitemBase, (data >> 10) & 0x3FU);
    addByte(l_WorkitemBase, (data >> 18) & 0x3FU);
    addByte(l_WorkitemBase, (data >> 26) & 0x3FU);
}

__kernel __attribute__((reqd_work_group_size(HISTOGRAM64_WORKGROUP_SIZE, 1, 1)))
void histogram64(
    __global uint *d_PartialHistograms,
    __global data_t *d_Data,
    uint dataCount
){

    const uint lPos = 
        ( (get_local_id(0) & ~(LOCAL_MEMORY_BANKS * 4 - 1)) << 0 ) |
        ( (get_local_id(0) &  (LOCAL_MEMORY_BANKS     - 1)) << 2 ) |
        ( (get_local_id(0) &  (LOCAL_MEMORY_BANKS * 3    )) >> 4 );

    __local uchar l_Hist[HISTOGRAM64_WORKGROUP_SIZE * HISTOGRAM64_BIN_COUNT];
    __local uchar *l_WorkitemBase = l_Hist + lPos;

    for(uint i = 0; i < (HISTOGRAM64_BIN_COUNT / 4); i++)
        ((__local uint *)l_Hist)[lPos + i * HISTOGRAM64_WORKGROUP_SIZE] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint pos = get_global_id(0); pos < dataCount; pos += get_global_size(0)){
        data_t data = d_Data[pos];
        addWord(l_WorkitemBase, data.x);
        addWord(l_WorkitemBase, data.y);
        addWord(l_WorkitemBase, data.z);
        addWord(l_WorkitemBase, data.w);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(get_local_id(0) < HISTOGRAM64_BIN_COUNT){
        __local uchar *l_HistBase = l_Hist + UMUL(get_local_id(0), HISTOGRAM64_WORKGROUP_SIZE);

        uint sum = 0;
        uint pos = 4 * (get_local_id(0) & (LOCAL_MEMORY_BANKS - 1));
        for(uint i = 0; i < (HISTOGRAM64_WORKGROUP_SIZE / 4); i++){
            sum += 
                l_HistBase[pos + 0] + 
                l_HistBase[pos + 1] + 
                l_HistBase[pos + 2] + 
                l_HistBase[pos + 3];
            pos = (pos + 4) & (HISTOGRAM64_WORKGROUP_SIZE - 1);
        }

        d_PartialHistograms[get_group_id(0) * HISTOGRAM64_BIN_COUNT + get_local_id(0)] = sum;
    }
}

__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void mergeHistogram64(
    __global uint *d_Histogram,
    __global uint *d_PartialHistograms,
    uint histogramCount
){
    __local uint l_Data[MERGE_WORKGROUP_SIZE];

    uint sum = 0;
    for(uint i = get_local_id(0); i < histogramCount; i += MERGE_WORKGROUP_SIZE)
        sum += d_PartialHistograms[get_group_id(0) + i * HISTOGRAM64_BIN_COUNT];
    l_Data[get_local_id(0)] = sum;

    for(uint stride = MERGE_WORKGROUP_SIZE / 2; stride > 0; stride >>= 1){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(get_local_id(0) < stride)
            l_Data[get_local_id(0)] += l_Data[get_local_id(0) + stride];
    }

    if(get_local_id(0) == 0)
        d_Histogram[get_group_id(0)] = l_Data[0];
}
