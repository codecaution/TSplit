#include "gpu_runtime.h"

__global__ void relu_kernel(float *input, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size) return;
    output[ind] = max(input[ind], 0.);
  }
  
int DLGpuRelu(const DLArrayHandle input, DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
    /* TODO: Your code here */
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
      size *= input->shape[i];
    }
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
        relu_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(input_data, output_data, size);
      else
        relu_kernel<<<blocks, threads>>>(input_data, output_data, size);
  
      float elapsedTime;
      cudaEventCreate(&stop);
      cudaEventRecord(stop,0);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&elapsedTime, start,stop);
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      p->time = elapsedTime;      
    }else{
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
        relu_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(input_data, output_data, size);
      else
        relu_kernel<<<blocks, threads>>>(input_data, output_data, size);
  
    }
    return 0;
}
  
__global__ void relu_grad_kernel(const float *input, const float *in_grad, float *output, size_t size) {
    size_t ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= size) return;
    float s = 0;
    if (input[ind] > 0) s = 1;
    if (input[ind] < 0) s= -1;
    output[ind] = (s + 1) * in_grad[ind] * 0.5;
}
  
int DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
                      DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
    size_t size = 1;
    for (index_t i = 0; i < input->ndim; i++) {
      size *= input->shape[i];
    }
    if(p != NULL){
      int size_input = 1, size_output = 1, size_in_grad = 1;
      for(int i = 0; i < input -> ndim; i++)
          size_input *= input -> shape[i];
      for(int i = 0; i < in_grad -> ndim; i++)
          size_in_grad *= in_grad -> shape[i];
      for(int i = 0; i < output -> ndim; i++)
          size_output *= output -> shape[i];
      p -> input_memory = 1.0 * (size_input + size_in_grad) * sizeof(float) / 1024 / 1024;
      p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
      p -> workspace_memory = 0;
        // Insert the begin and end event.
      // cudaEvent_t start, stop;
      // cudaEventCreate(&start);
      // cudaEventRecord(start,0);

      dim3 blocks;
      dim3 threads;
      const float *input_data = (const float *)input->data;
      const float *in_grad_data = (const float *)in_grad->data; 
      float *output_data = (float *)output->data;
      if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
      } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
      }
      if (stream_handle)
        relu_grad_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(input_data, in_grad_data, output_data, size);
      else
        relu_grad_kernel<<<blocks, threads>>>(input_data, in_grad_data, output_data, size);
      // float elapsedTime;
      // cudaEventCreate(&stop);
      // cudaEventRecord(stop,0);
      // cudaEventSynchronize(stop);
      // cudaEventElapsedTime(&elapsedTime, start,stop);
      // cudaEventDestroy(start);
      // cudaEventDestroy(stop);
      // p->time = elapsedTime;
    }else{
      dim3 blocks;
      dim3 threads;
      const float *input_data = (const float *)input->data;
      const float *in_grad_data = (const float *)in_grad->data; 
      float *output_data = (float *)output->data;
      if (size <= 1024) {
        threads.x = size;
        blocks.x = 1;
      } else {
        threads.x = 1024;
        blocks.x = (size + 1023) / 1024;
      }
      if (stream_handle)
        relu_grad_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(input_data, in_grad_data, output_data, size);
      else
        relu_grad_kernel<<<blocks, threads>>>(input_data, in_grad_data, output_data, size);  
    }
    return 0;
}