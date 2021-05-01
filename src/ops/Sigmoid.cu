#include "gpu_runtime.h"

__global__ void sigmoid_kernel(float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size) return;
    output[ind] = 1.0/(1.0+1.0/exp(input[ind]));
  }
  
int DLGpuSigmoid(const DLArrayHandle input, DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
    /* TODO: Your code here */
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
      size *= input->shape[i];
    }
    dim3 blocks;
    dim3 threads;
    float *input_data = (float *)input->data;
    float *output_data = (float *)output->data;
    if (size <= 1024) {
      threads.x = size;
      blocks.x = 1;
    } else {
      threads.x = 1024;
      blocks.x = (size + 1023) / 1024;
    }
    if (stream_handle)
      sigmoid_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(input_data, output_data, size);
    else
      sigmoid_kernel<<<blocks, threads>>>(input_data, output_data, size);
    if(p != NULL){
      int size_input = 1, size_output = 1;
      for(int i = 0; i < input -> ndim; i++)
        size_input *= input -> shape[i];
      for(int i = 0; i < output -> ndim; i++)
        size_output *= output -> shape[i];
      p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
      p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
      p -> workspace_memory = 0;
    }
    return 0;
}