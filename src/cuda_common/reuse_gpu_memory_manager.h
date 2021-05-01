#include "dlarray.h"
#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

class ReuseDLMemoryManager{
public:
    //unfinished
    size_t GpuMemoryCapacity, UsedMemory, AvailableMemory;
    void* base_address;
    void CudeSetDevice(DLContext ctx);
    void* CudaApplyMemory(size_t memory_size);
    ReuseDLMemoryManager(DLContext ctx);
    void* DLMemoryMalloc(size_t memory_size);
    void DLMemoryFree(void* ptr, size_t memory_size=0);
    int DLMemoryPrintState();
    bool DLMemoryTry(size_t memory_size);
    void DLMemoryUsage(size_t *used_size, size_t *unused_unavailable_size, size_t *unused_available_size);
};