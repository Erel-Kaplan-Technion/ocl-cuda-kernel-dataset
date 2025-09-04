#include "scan_common.h"

extern "C" void scanExclusiveHost(uint *dst, uint *src, uint batchSize, uint arrayLength)
{
    for (uint i = 0; i < batchSize; i++, src += arrayLength, dst += arrayLength) {
        dst[0] = 0;

        for (uint j = 1; j < arrayLength; j++)
            dst[j] = src[j - 1] + dst[j - 1];
    }
}
