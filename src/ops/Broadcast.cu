#include "gpu_runtime.h"


__global__ void broadcast_to_kernel(const float *input_data,float *output_data,size_t input_size,size_t output_size){
  size_t id = blockIdx.x * blockDim.x +threadIdx.x;
  if(id >= output_size)return ;
  output_data[id] = input_data[id%input_size];
}
  
int DLGpuBroadcastTo(const DLArrayHandle input, DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){

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

    for(index_t i = 0; i < input->ndim; i++){
      assert((input->shape[i]) == (output->shape[i+1]));
    }
    size_t input_size = 1;
    for(index_t i = 0;i < input->ndim; i++){
      input_size *= input->shape[i];
    }
    size_t output_size = input_size * (output->shape[0]);
    size_t BLOCKS = (output_size + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;
    if (stream_handle)
    {
      cudaStream_t *s = (cudaStream_t*)(stream_handle->handle);
      broadcast_to_kernel<<<BLOCKS,THREADS_PER_BLOCK, 0, *s>>>((const float*)(input->data),(float*)(output->data),input_size,output_size);
      //broadcast_to_kernel<<<BLOCKS,THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>((const float*)(input->data),(float*)(output->data),input_size,output_size);
    }
    else
      broadcast_to_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>((const float*)(input->data),(float*)(output->data),input_size,output_size);
  
    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    p->time = elapsedTime; 
  }else{
    for(index_t i = 0; i < input->ndim; i++){
      assert((input->shape[i]) == (output->shape[i+1]));
    }
    size_t input_size = 1;
    for(index_t i = 0;i < input->ndim; i++){
      input_size *= input->shape[i];
    }
    size_t output_size = input_size * (output->shape[0]);
    size_t BLOCKS = (output_size + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;
    if (stream_handle)
    {
      cudaStream_t *s = (cudaStream_t*)(stream_handle->handle);
      broadcast_to_kernel<<<BLOCKS,THREADS_PER_BLOCK, 0, *s>>>((const float*)(input->data),(float*)(output->data),input_size,output_size);
      //broadcast_to_kernel<<<BLOCKS,THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>((const float*)(input->data),(float*)(output->data),input_size,output_size);
    }
    else
      broadcast_to_kernel<<<BLOCKS,THREADS_PER_BLOCK>>>((const float*)(input->data),(float*)(output->data),input_size,output_size);  
  }
  return 0;
}
