#include "reuse_gpu_memory_manager.h"

ReuseDLMemoryManager *MemoryPool = NULL;

#define CUDA_CALL(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void ReuseDLMemoryManager::CudeSetDevice(DLContext ctx){
    CUDA_CALL(cudaSetDevice(ctx.device_id));
}

void* ReuseDLMemoryManager::CudaApplyMemory(size_t memory_size){
    void *ret;
    CUDA_CALL(cudaMalloc(&ret, memory_size));
    return ret;
}

bool ReuseDLMemoryManager::DLMemoryTry(size_t memory_size)
{
// todo: this is only a test
    if(memory_size > AvailableMemory) return false;
    return true;
}

void* ReuseDLMemoryManager::DLMemoryMalloc(size_t memory_size){
    assert(memory_size <= AvailableMemory);
    AvailableMemory -= memory_size;
    UsedMemory += memory_size;
//    cout << base_address << endl;
    return base_address;
}

ReuseDLMemoryManager::ReuseDLMemoryManager(DLContext ctx){
    CudeSetDevice(ctx);
    size_t avail, total;
    cudaMemGetInfo(&avail, &total);
    // avail = 12LL * 1024 * 1024 * 1024;
    GpuMemoryCapacity = avail - 600LL * 1024 * 1024;
    AvailableMemory = GpuMemoryCapacity;
    UsedMemory = 0;

    void* ptr = CudaApplyMemory(GpuMemoryCapacity);
    if(ptr == NULL){
        printf("DLMemoryManager apply for initial memory of gpu false. ");
        assert(1 == -1);
    }
    base_address = ptr;
//    cout << base_address << " " << (avail - 300 * 1024 * 1024) / 1024.0 / 1024.0 << endl;
}

void ReuseDLMemoryManager::DLMemoryFree(void* ptr, size_t memory_size)
{
    if(memory_size == 0){
        printf("You are using WRONG API or you are trying to free a space with memory size Zero. You have to send memory size when you want to Free data.");
        assert(1==-1);
    }
    UsedMemory -= memory_size;
    AvailableMemory += memory_size;
}

int ReuseDLMemoryManager::DLMemoryPrintState()
{
   cout << "Used Memory is: "<< UsedMemory / 1024.0 / 1024.0 \
    << " MB, Available Memory is: " << AvailableMemory / 1024.0 / 1024.0 << "MB" << endl;
   return -1;
}

void ReuseDLMemoryManager::DLMemoryUsage(size_t *used_size, size_t *unused_unavailable_size, size_t *unused_available_size)
{
    (*used_size) = UsedMemory;
    (*unused_unavailable_size) = 0;
    (*unused_available_size) = AvailableMemory;
}
