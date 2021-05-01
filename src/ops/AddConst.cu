#include "gpu_runtime.h"

__global__ void add_const_kernel(const float *input, float *output, float value, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size) return;
    output[ind] = input[ind] + value;
  }
  
int DLGpuMatrixElementwiseAddByConst(const DLArrayHandle input, float val,
                                       DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
                                           
    if(p != NULL){
        int size_input = 1, size_output = 1;
        for(int i = 0; i < input -> ndim; i++)
            size_input *= input -> shape[i];
        for(int i = 0; i < output -> ndim; i++)
            size_output *= output -> shape[i];
        p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
        p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
        p -> workspace_memory = 0;
        // Insert the begin and end event.
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);

        size_t size = 1;
        for (index_t i = 0; i < input->ndim; i++) {
          size *= input->shape[i];
        }
        dim3 blocks;
        dim3 threads;
        float *output_data = (float *)output->data;
        const float *input_data = (const float *)input->data;
        if (size <= 1024) {
          threads.x = size;
          blocks.x = 1;
        } else {
          threads.x = 1024;
          blocks.x = (size + 1023) / 1024;
        }
        if (stream_handle)
          add_const_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(input_data, output_data, val, size);
        else
          add_const_kernel<<<blocks, threads>>>(input_data, output_data, val, size);

        float elapsedTime;
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        p->time = elapsedTime;
    }
    else{
      size_t size = 1;
      for (index_t i = 0; i < input->ndim; i++) {
        size *= input->shape[i];
      }
      dim3 blocks;
      dim3 threads;
      float *output_data = (float *)output->data;
      const float *input_data = (const float *)input->data;
      if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
      } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
      }
      if (stream_handle)
        add_const_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(input_data, output_data, val, size);
      else
        add_const_kernel<<<blocks, threads>>>(input_data, output_data, val, size);
    }
    return 0;
}