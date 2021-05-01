/*!
 *  Copyright (c) 2017 by Contributors
 * \file c_runtime_api.cc
 * \brief Device specific implementations
 */
#include "c_runtime_api.h"
#include "cpu_device_api.h"
#include "cuda_device_api.h"
#include "runtime_base.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <thread>

namespace dlsys {
namespace runtime {

class DeviceAPIManager {
public:
  static const int kMaxDeviceAPI = 8;
  // Get API
  static DeviceAPI *Get(DLContext ctx) {
    return Global()->GetAPI(ctx.device_type);
  }

private:
  std::array<DeviceAPI *, kMaxDeviceAPI> api_;
  DeviceAPIManager() {
    std::fill(api_.begin(), api_.end(), nullptr);
    static CPUDeviceAPI cpu_device_api_inst;
    api_[kCPU] = static_cast<DeviceAPI *>(&cpu_device_api_inst);
    #ifdef DEVICE_GPU
    static CUDADeviceAPI gpu_device_api_inst;
    api_[kGPU] = static_cast<DeviceAPI *>(&gpu_device_api_inst);
    #endif
  }
  // Get global static variable.
  static DeviceAPIManager *Global() {
    static DeviceAPIManager inst;
    return &inst;
  }
  // Get API.
  DeviceAPI *GetAPI(DLDeviceType type) {
    if (api_[type] == nullptr) {
      std::cout<< "Device type:"<<type <<std::endl;
      std::cerr << "Device API not supported" << std::endl;
      exit(EXIT_FAILURE);
    }
    return api_[type];
  }
};

inline DLArray *DLArrayCreate_() {
  DLArray *arr = new DLArray();
  arr->shape = nullptr;
  arr->ndim = 0;
  arr->data = nullptr;
  return arr;
}

inline void DLArrayFree_(DLArray *arr, int MEMORY_MANAGE_RULE) {
  size_t memory_size = 1;
  for(int i = 0; i < arr->ndim; i++)
    memory_size *= (arr -> shape)[i];
  memory_size = memory_size * 4;
  if (arr != nullptr) {
    // ok to delete nullptr
    delete[] arr->shape;
    if (arr->data != nullptr) {
      DeviceAPIManager::Get(arr->ctx)->FreeDataSpace(arr->ctx, arr->data, MEMORY_MANAGE_RULE, memory_size);
    }
  }
  delete arr;
}

inline void DLArrayFreeSelf_(DLArray *arr, int MEMORY_MANAGE_RULE) {
  size_t memory_size = 1;
  for(int i = 0; i < arr->ndim; i++)
    memory_size *= (arr -> shape)[i];
  memory_size = memory_size * 4;
  if (arr->data != nullptr) {
    DeviceAPIManager::Get(arr->ctx)->FreeDataSpace(arr->ctx, arr->data, MEMORY_MANAGE_RULE, memory_size);
  }
}

inline size_t GetDataSize(DLArray *arr) {
  size_t size = 1;
  for (index_t i = 0; i < arr->ndim; ++i) {
    size *= arr->shape[i];
  }
  // assume 32-bit float
  size *= 4;
  return size;
}

inline size_t GetDataAlignment(DLArray *arr) {
  // assume 32-bit float
  return 8;
}

} // namespace runtime
} // namespace dlsys

using namespace dlsys::runtime;

int ProfilerAlloc(ProfilerHandle* p)
{
    *p = new Profiler;
//    std::cout << "Attention:" << (*p) -> time << std::endl;
    return 0;
}

//int DLArrayAlloc(const index_t *shape, index_t ndim, DLContext ctx,
//                 DLArrayHandle *out) {
//  DLArray *arr = nullptr;
//  API_BEGIN();
//  // shape
//  arr = DLArrayCreate_();
//  // ndim
//  arr->ndim = ndim;
//  index_t *shape_copy = new index_t[ndim];
//  std::copy(shape, shape + ndim, shape_copy);
//  arr->shape = shape_copy;
//  // ctx
//  arr->ctx = ctx;
//  size_t size = GetDataSize(arr);
//  size_t alignment = GetDataAlignment(arr);
//  arr->data = DeviceAPIManager::Get(ctx)->AllocDataSpace(ctx, size, alignment);
//  *out = arr;
//  API_END_HANDLE_ERROR(DLArrayFree_(arr));
//}

int DLArrayAlloc(const index_t *shape, index_t ndim, DLContext ctx,
                 DLArrayHandle *out, int MEMORY_MANAGE_RULE) {
  DLArray *arr = nullptr;
  API_BEGIN();
  // shape
  arr = DLArrayCreate_();
  // ndim
  arr->ndim = ndim;
  index_t *shape_copy = new index_t[ndim];
  std::copy(shape, shape + ndim, shape_copy);
  arr->shape = shape_copy;
  // ctx
  arr->ctx = ctx;
  size_t size = GetDataSize(arr);
  size_t alignment = GetDataAlignment(arr);
  arr->data = DeviceAPIManager::Get(ctx)->AllocDataSpace(ctx, size, alignment, MEMORY_MANAGE_RULE);
  *out = arr;
  API_END_HANDLE_ERROR(DLArrayFree_(arr, MEMORY_MANAGE_RULE));
}

int DLArrayAllocSelf(DLArrayHandle handle, int MEMORY_MANAGE_RULE){
  size_t size = GetDataSize(handle);
  size_t alignment = GetDataAlignment(handle);
  handle->data = DeviceAPIManager::Get(handle->ctx)->AllocDataSpace(handle->ctx, size, alignment, MEMORY_MANAGE_RULE);
  return 0;
}

int DLArrayReuseAlloc(DLContext ctx, int MEMORY_MANAGE_RULE, 
                        const index_t *input_shape, index_t input_ndim, DLArrayHandle *input,
                        const index_t *output_shape, index_t output_ndim, DLArrayHandle *output,
                        int pieces, int overlap_pieces){
  void *arr_data = nullptr;
  DLArray *input_arr = nullptr;
  DLArray *output_arr = nullptr;

  size_t input_size = 1;
  for (index_t i = 0; i < input_ndim; i++) {
    input_size *= input_shape[i];
  }
  input_size *= 4;
  
  size_t output_size = 1;
  for (index_t i = 0; i < output_ndim; i++) {
    output_size *= output_shape[i];
  } 
  output_size *= 4;

  size_t total_size = output_size + input_size / pieces * (pieces - overlap_pieces);
  size_t alignment = 8;
  arr_data = DeviceAPIManager::Get(ctx)->AllocDataSpace(ctx, total_size, alignment, MEMORY_MANAGE_RULE);

  input_arr = DLArrayCreate_();
  input_arr->ndim = input_ndim;
  index_t *input_shape_copy = new index_t[input_ndim];
  std::copy(input_shape, input_shape + input_ndim, input_shape_copy);
  input_arr->shape = input_shape_copy;
  input_arr->ctx = ctx;

  input_arr->data = (float*)arr_data + (total_size - input_size)/4;

  output_arr = DLArrayCreate_();
  output_arr->ndim = output_ndim;
  index_t *output_shape_copy = new index_t[output_ndim];
  std::copy(output_shape, output_shape + output_ndim, output_shape_copy);
  output_arr->shape = output_shape_copy;
  output_arr->ctx = ctx;

  output_arr->data = arr_data;

  *input = input_arr;
  *output = output_arr;
  // API_END_HANDLE_ERROR(DLArrayFree_(arr_data, MEMORY_MANAGE_RULE));
  return 0;
}

int DLArraySliceForMicroTensor(DLArrayHandle input, DLArrayHandle *output, int total_number, int index, int dimension){
  DLArray *output_arr = nullptr;
  output_arr = DLArrayCreate_();
  output_arr->ndim = input->ndim;
  size_t input_size = 1;
  for(index_t i = 0; i < (input->ndim); i++){
    input_size *= input->shape[i];
  }
  // input_size *= 4;
  index_t *output_shape = new index_t[output_arr->ndim];
  for(index_t i = 0; i < (output_arr->ndim); i++){
    output_shape[i] = input->shape[i];
  }
  // printf("dimension: %d\n", dimension);
  if(dimension == 0){
    index_t N = output_shape[0];
    index_t K = N % total_number;
    output_shape[0] /= total_number;
    index_t len = input_size / N;

    index_t ptr_size = 0;
    for(index_t i = 1; i <= index; i++){
      if(i <= K){
        ptr_size += (N/total_number + 1) * len;
      }
      else{
        ptr_size += (N/total_number)*len;
      }
    }
    if(index < K){
      output_shape[0]++;
    }
    output_arr->ctx = input->ctx;
    output_arr->shape = output_shape;
    output_arr->data = ((float *)input->data) + ptr_size;
  }else{
    index_t C = output_shape[1];
    index_t K = C % total_number;
    output_shape[1] /= total_number;
    index_t len = input_size / C;

    index_t ptr_size = 0;
    for(index_t i = 1; i <= index; i++){
      if(i <= K){
        ptr_size += (C/total_number + 1)*len;
      }
      else{
        ptr_size += (C/total_number)*len;
      }
    }
    if(index < K){
      output_shape[1]++;
    }
    output_arr->ctx = input->ctx;
    output_arr->shape = output_shape;
    output_arr->data = ((float *)input->data) + ptr_size;
  }
  *output = output_arr;
  return 0;
}

int DLArrayFree(DLArrayHandle handle, int MEMORY_MANAGE_RULE) {
  API_BEGIN();
  DLArray *arr = handle;
  DLArrayFree_(arr, MEMORY_MANAGE_RULE);
  API_END();
}

int DLArrayFreeSelf(DLArrayHandle handle, int MEMORY_MANAGE_RULE) {
  API_BEGIN();
  DLArray *arr = handle;
  DLArrayFreeSelf_(arr, MEMORY_MANAGE_RULE);
  API_END();
}


int DLArrayCopyFromTo(DLArrayHandle from, DLArrayHandle to,
                      DLStreamHandle stream) {
  API_BEGIN();
  size_t from_size = GetDataSize(from);
  size_t to_size = GetDataSize(to);
  // The size must exactly match
  assert(from_size == to_size);
  DLContext ctx = from->ctx;
  if (ctx.device_type == kCPU) {
    ctx = to->ctx;
  } else {
    // Can not copy across different ctx types directly
    assert((to->ctx.device_type == kCPU) ||
           (to->ctx.device_type == from->ctx.device_type));
  }
  DeviceAPIManager::Get(ctx)->CopyDataFromTo(from->data, to->data, from_size,
                                             from->ctx, to->ctx, stream);
  API_END();
}

void DLMemoryManager_GPU_Usage(size_t *used_size, size_t *unused_unavailable_size, size_t *unused_available_size)
{
//    std::cout << used_size << " " << unused_available_size << " " << unused_unavailable_size << std::endl;
    if(MemoryPool == NULL){
        std::cout << "Error! MemoryPool is not available! " << std::endl;
        assert(1 == -1);
    }
//    std::cout << MemoryPool << std::endl;
    MemoryPool -> DLMemoryUsage(used_size, unused_unavailable_size, unused_available_size);
}

void DLMemoryManager_Try_to_Malloc(size_t memory_size, void* p)
{
    if(MemoryPool == NULL){
        std::cout<< "Error! MemoryPool is not available! " << std::endl;
        assert(1 == -1);
    }
    if(MemoryPool -> DLMemoryTry(memory_size))
        *((int*)p) = 1;
    else *((int*)p) = 0;
}

// void DLMemoryManager_PrintState(void *t, int type)
// {
//     if(MemoryPool == NULL){
//         std::cout<< "Error! MemoryPool is not available! " << std::endl;
//         assert(1 == -1);
//     }
//     int *p = (int*)t;
//     *p = MemoryPool -> DLMemoryPrintState(type);
// }
void DLMemoryManager_PrintState(void *t, int type)
{
    if(MemoryPool == NULL){
        std::cout<< "Error! MemoryPool is not available! " << std::endl;
        assert(1 == -1);
    }
    int *p = (int*)t;
    *p = MemoryPool -> DLMemoryPrintState();
}