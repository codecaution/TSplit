/*!
 *  Copyright (c) 2017 by Contributors
 * \file cpu_device_api.cc
 */
#include "cpu_device_api.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#ifdef DEVICE_GPU
#include <cuda_runtime.h>
#define CUDA_CALL(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)
#endif

namespace dlsys
{
namespace runtime
{
void *CPUDeviceAPI::AllocDataSpace(DLContext ctx, size_t size,
                                   size_t alignment, int MEMORY_MANAGE_RULE)
{
  // std::cout << "allocating cpu data" << std::endl;
  void *ptr;
  #ifdef DEVICE_GPU
  CUDA_CALL(cudaMallocHost((void**)&ptr, size));
  #else
  int ret = posix_memalign(&ptr, alignment, size);
  if (ret != 0)
    throw std::bad_alloc();
  #endif
  return ptr;
}

void CPUDeviceAPI::FreeDataSpace(DLContext ctx, void *ptr,int MEMORY_MANAGE_RULE, size_t memory_size)
{
  #ifdef DEVICE_GPU
  cudaFreeHost(ptr);
  #else
  free(ptr);
  #endif
}

void CPUDeviceAPI::CopyDataFromTo(const void *from, void *to, size_t size,
                                  DLContext ctx_from, DLContext ctx_to,
                                  DLStreamHandle stream)
{
  // std::cout << "copying cpu data" << std::endl;
  memcpy(to, from, size);
}

void CPUDeviceAPI::StreamSync(DLContext ctx, DLStreamHandle stream) {}

} // namespace runtime
} // namespace dlsys
