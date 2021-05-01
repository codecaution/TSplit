#include "gpu_runtime.h"

__global__ void conv2d_reduce_kernel(const float * input_data, float *output_data, size_t input_size, size_t output_size, size_t batch_size){
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= output_size) return;
    float temp = 0;
    for(int i = 0; i < batch_size; i++){
      for ( int j = 0; j < input_size; j++){
          temp += input_data[i * input_size * output_size + id * input_size + j];
      } 
    }
    output_data[id] = temp;
  }
  //  a naive type!!!
  int DLGpuConv2d_reduce_sum(const DLArrayHandle input_x, DLArrayHandle output_y, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
    assert(input_x -> shape[1] == output_y -> shape[0]);
    const float *input_data = (const float *) input_x -> data;
    float* output_data = (float *) output_y ->data;
    size_t batch_size = input_x -> shape[0];
    size_t input_size = input_x -> shape[2] * input_x -> shape[3];
    size_t output_size = output_y ->shape[0];
  
    size_t BLOCKS = (output_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (stream_handle)
      conv2d_reduce_kernel<<<BLOCKS, THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>(input_data, output_data, input_size, output_size, batch_size);
    else
      conv2d_reduce_kernel<<<BLOCKS, THREADS_PER_BLOCK>>>(input_data, output_data, input_size, output_size, batch_size);
    if(p != NULL){
        int size_input = 1, size_output = 1;
        for(int i = 0; i < input_x -> ndim; i++)
            size_input *= input_x -> shape[i];
        for(int i = 0; i < output_y -> ndim; i++)
            size_output *= output_y -> shape[i];
        p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
        p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
        p -> workspace_memory = 0;
    }
    return 0;
  }