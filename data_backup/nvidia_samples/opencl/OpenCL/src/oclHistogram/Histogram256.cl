#define HISTOGRAM256_BIN_COUNT 256

#define      UINT_BITS 32U
#define      WARP_SIZE (1U << LOG2_WARP_SIZE)

#define HISTOGRAM256_WORKGROUP_SIZE (WARP_COUNT * WARP_SIZE)

#define HISTOGRAM256_WORKGROUP_MEMORY (WARP_COUNT * HISTOGRAM256_BIN_COUNT)

#define TAG_MASK ( (1U << (UINT_BITS - LOG2_WARP_SIZE)) - 1U )

inline void addByte(volatile __local uint *l_WarpHist, uint data, uint tag){
    uint count;
    do{
        count = l_WarpHist[data] & TAG_MASK;
        count = tag | (count + 1);
        l_WarpHist[data] = count;
    }while(l_WarpHist[data] != count);
}

inline void addWord(volatile __local uint *l_WarpHist, uint data, uint tag){
    addByte(l_WarpHist, (data >>  0) & 0xFFU, tag);
    addByte(l_WarpHist, (data >>  8) & 0xFFU, tag);
    addByte(l_WarpHist, (data >> 16) & 0xFFU, tag);
    addByte(l_WarpHist, (data >> 24) & 0xFFU, tag);
}

__kernel __attribute__((reqd_work_group_size(HISTOGRAM256_WORKGROUP_SIZE, 1, 1)))
void histogram256(
    __global uint *d_PartialHistograms,
    __global uint *d_Data,
    uint dataCount
){
    //Per-warp substorage storage
    __local uint l_Hist[WARP_COUNT * HISTOGRAM256_BIN_COUNT];
    __local uint *l_WarpHist = l_Hist + (get_local_id(0) >> LOG2_WARP_SIZE) * HISTOGRAM256_BIN_COUNT;

    //Clear shared memory storage for current threadblock before processing
    for(uint i = 0; i < (HISTOGRAM256_BIN_COUNT / WARP_SIZE); i++)
        l_Hist[get_local_id(0) + i  * (WARP_COUNT * WARP_SIZE)] = 0;

    const uint tag =  get_local_id(0) << (32 - LOG2_WARP_SIZE);

    //Read through the entire input buffer, build per-warp histograms
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint pos = get_global_id(0); pos < dataCount; pos += get_global_size(0)){
        uint data = d_Data[pos];
        addWord(l_WarpHist, data, tag);
    }

    //Per-block histogram reduction
    barrier(CLK_LOCAL_MEM_FENCE);
    for(uint pos = get_local_id(0); pos < HISTOGRAM256_BIN_COUNT; pos += HISTOGRAM256_WORKGROUP_SIZE){
        uint sum = 0;

        for(uint i = 0; i < WARP_COUNT; i++)
            sum += l_Hist[pos + i * HISTOGRAM256_BIN_COUNT] & TAG_MASK;

        d_PartialHistograms[get_group_id(0) * HISTOGRAM256_BIN_COUNT + pos] = sum;
    }
}

__kernel __attribute__((reqd_work_group_size(MERGE_WORKGROUP_SIZE, 1, 1)))
void mergeHistogram256(
    __global uint *d_Histogram,
    __global uint *d_PartialHistograms,
    uint histogramCount
){
    __local uint l_Data[MERGE_WORKGROUP_SIZE];

    uint sum = 0;
    for(uint i = get_local_id(0); i < histogramCount; i += MERGE_WORKGROUP_SIZE)
        sum += d_PartialHistograms[get_group_id(0) + i * HISTOGRAM256_BIN_COUNT];
    l_Data[get_local_id(0)] = sum;

    for(uint stride = MERGE_WORKGROUP_SIZE / 2; stride > 0; stride >>= 1){
        barrier(CLK_LOCAL_MEM_FENCE);
        if(get_local_id(0) < stride)
            l_Data[get_local_id(0)] += l_Data[get_local_id(0) + stride];
    }

    if(get_local_id(0) == 0)
        d_Histogram[get_group_id(0)] = l_Data[0];
}
