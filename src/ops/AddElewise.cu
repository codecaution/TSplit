#include "gpu_runtime.h"

__global__ void ele_add_kernel(const float *matA, const float *matB, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size) return;
    output[ind] = matA[ind] + matB[ind];
  }
  
  int DLGpuMatrixElementwiseAdd(const DLArrayHandle matA,
                                const DLArrayHandle matB, DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  if(p != NULL){
    int size_a = 1, size_b = 1, size_c = 1;
    for(int i = 0; i < matA -> ndim; i++)
        size_a *= matA -> shape[i];
    for(int i = 0; i < matB -> ndim; i++)
        size_b *= matB -> shape[i];
    for(int i = 0; i < output -> ndim; i++)
        size_c *= output -> shape[i];
    p -> input_memory = 1.0 * (size_a  + size_b) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_c * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    int dev_id = (matA->ctx).device_id;
    cudaSetDevice(dev_id);
    size_t size = 1;
    for (index_t i = 0; i< matA->ndim; i++) {
      size *= matA->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)output->data;
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    if (size <= 1024) {
      threads.x = size;
      blocks.x = 1;
    } else {
      threads.x = 1024;
      blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
      ele_add_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(matA_data, matB_data, output_data, size);
    else
      ele_add_kernel<<<blocks, threads>>>(matA_data, matB_data, output_data, size);

    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    p->time = elapsedTime;
  }else{
    int dev_id = (matA->ctx).device_id;
    cudaSetDevice(dev_id);
    size_t size = 1;
    for (index_t i = 0; i< matA->ndim; i++) {
      size *= matA->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *output_data = (float *)output->data;
    const float *matA_data = (const float *)matA->data;
    const float *matB_data = (const float *)matB->data;
    if (size <= 1024) {
      threads.x = size;
      blocks.x = 1;
    } else {
      threads.x = 1024;
      blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
      ele_add_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(matA_data, matB_data, output_data, size);
    else
      ele_add_kernel<<<blocks, threads>>>(matA_data, matB_data, output_data, size);
  }
    return 0;
}