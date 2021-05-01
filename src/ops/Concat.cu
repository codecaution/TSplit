#include "gpu_runtime.h"


__global__ void concat_kernel(const int nthreads, const float* in_data,
    const bool forward, const int num_concats, const int concat_size,
    const int top_concat_axis, const int bottom_concat_axis,
    const int offset_concat_axis, float* out_data) {
    CUDA_KERNEL_LOOP(index, nthreads) {
    const int total_concat_size = concat_size * bottom_concat_axis;
    const int concat_num = index / total_concat_size;
    const int concat_index = index % total_concat_size;
    const int top_index = concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    if (forward) {
      out_data[top_index] = in_data[index];
    } else {
      out_data[index] = in_data[top_index];
    }
  }
}

int DLGpuConcat(const DLArrayHandle input_x, const DLArrayHandle input_y, DLArrayHandle output, int axis = 0, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  assert(input_x -> ndim == input_y -> ndim);
  assert(input_y -> ndim == output -> ndim);
  int now_ndim = input_x -> ndim;
  for(int i = 0; i < now_ndim; i++){
    if(i != axis){
      assert(input_x -> shape[i] == input_y -> shape[i]);
      assert(input_y -> shape[i] == output -> shape[i]);
    }
    else{
      assert(input_x -> shape[i] + input_y -> shape[i] == output -> shape[i]);
    }
  }
  if(p != NULL){
    int size_a = 1, size_b = 1, size_c = 1;
    for(int i = 0; i < input_x -> ndim; i++)
        size_a *= input_x -> shape[i];
    for(int i = 0; i < input_y -> ndim; i++)
        size_b *= input_y -> shape[i];
    for(int i = 0; i < output -> ndim; i++)
        size_c *= output -> shape[i];
    p -> input_memory = 1.0 * (size_a  + size_b) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_c * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    int concat_size = 1;
    for(int i = axis + 1; i < now_ndim; i++){
      concat_size *= input_x -> shape[i];
    }
    int num_concats = 1;
    for(int i = 0; i< axis; i++){
      num_concats *= input_x -> shape[i];
    }
    int concat_offset = 0;
    float *output_data = (float *)(output -> data);
    for(int i = 0; i < 2; i++){
      int input_concat_axis;
      const float *input_data;
      if(i == 0){ // input_x
        input_concat_axis = input_x -> shape[axis];
        input_data = (const float *)(input_x -> data);
      }
      else{  //input_y
        input_concat_axis = input_y -> shape[axis];
        input_data = (const float *)(input_y -> data);
      }
      const int input_concat_size = input_concat_axis * concat_size;
      const int nthreads = input_concat_size * num_concats;
      const int blocks = (nthreads + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;
      if (stream_handle)
        concat_kernel<<<blocks, THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>(nthreads, input_data, true, num_concats, concat_size,
          output -> shape[axis], input_concat_axis, concat_offset, output_data);
      else
        concat_kernel<<<blocks, THREADS_PER_BLOCK>>>(nthreads, input_data, true, num_concats, concat_size,
          output -> shape[axis], input_concat_axis, concat_offset, output_data);
      concat_offset += input_concat_axis;
    }
    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    p->time = elapsedTime;
  }else{
    int concat_size = 1;
    for(int i = axis + 1; i < now_ndim; i++){
      concat_size *= input_x -> shape[i];
    }
    int num_concats = 1;
    for(int i = 0; i< axis; i++){
      num_concats *= input_x -> shape[i];
    }
    int concat_offset = 0;
    float *output_data = (float *)(output -> data);
    for(int i = 0; i < 2; i++){
      int input_concat_axis;
      const float *input_data;
      if(i == 0){ // input_x
        input_concat_axis = input_x -> shape[axis];
        input_data = (const float *)(input_x -> data);
      }
      else{  //input_y
        input_concat_axis = input_y -> shape[axis];
        input_data = (const float *)(input_y -> data);
      }
      const int input_concat_size = input_concat_axis * concat_size;
      const int nthreads = input_concat_size * num_concats;
      const int blocks = (nthreads + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;
      if (stream_handle)
        concat_kernel<<<blocks, THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>(nthreads, input_data, true, num_concats, concat_size,
          output -> shape[axis], input_concat_axis, concat_offset, output_data);
      else
        concat_kernel<<<blocks, THREADS_PER_BLOCK>>>(nthreads, input_data, true, num_concats, concat_size,
          output -> shape[axis], input_concat_axis, concat_offset, output_data);
      concat_offset += input_concat_axis;
    }
  }
  return 0;
}

int DLGpuConcat_gradient(const DLArrayHandle output_gradient, DLArrayHandle input_gradient, int axis = 0, int id = 0, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  assert(output_gradient -> ndim == input_gradient -> ndim);
  if(p != NULL){
    int size_a = 1, size_b = 1;
    for(int i = 0; i < output_gradient -> ndim; i++)
        size_a *= output_gradient -> shape[i];
    for(int i = 0; i < input_gradient -> ndim; i++)
        size_b *= input_gradient -> shape[i];
    p -> input_memory = 1.0 * (size_a) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_b * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);
    int now_ndim = output_gradient -> ndim;
    int concat_offset = 0;
    for(int i = 0; i< now_ndim; i++){
      if(i!=axis){
        assert(input_gradient -> shape[i] == output_gradient -> shape[i]);
      }
      else{
        if(id == 1){
          concat_offset = (output_gradient -> shape[i]) - (input_gradient -> shape[i]);
        }
      }
    }
    int concat_size = 1;
    int num_concats = 1;
    for(int i = axis + 1; i < now_ndim; i++){
      concat_size *= output_gradient -> shape[i];
    }
    for(int i = 0; i< axis; i++){
      num_concats *= output_gradient -> shape[i];
    }
    const float * grad_out_data = (const float *)(output_gradient -> data);
    float * grad_in_data = (float *)(input_gradient -> data);
    const int input_concat_axis = input_gradient -> shape[axis];
    const int output_concat_axis = output_gradient -> shape[axis];
  
  
    const int input_concat_size = input_concat_axis * concat_size;
    const int nthreads = input_concat_size * num_concats;
    const int blocks = (nthreads + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;
    if (stream_handle)
      concat_kernel<<<blocks, THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>(nthreads, grad_out_data, false, num_concats, concat_size,
       output_concat_axis, input_concat_axis, concat_offset, grad_in_data);
    else
      concat_kernel<<<blocks, THREADS_PER_BLOCK>>>(nthreads, grad_out_data, false, num_concats, concat_size,
       output_concat_axis, input_concat_axis, concat_offset, grad_in_data);
  
    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    p->time = elapsedTime;
  }else{
    int now_ndim = output_gradient -> ndim;
    int concat_offset = 0;
    for(int i = 0; i< now_ndim; i++){
      if(i!=axis){
        assert(input_gradient -> shape[i] == output_gradient -> shape[i]);
      }
      else{
        if(id == 1){
          concat_offset = (output_gradient -> shape[i]) - (input_gradient -> shape[i]);
        }
      }
    }
    int concat_size = 1;
    int num_concats = 1;
    for(int i = axis + 1; i < now_ndim; i++){
      concat_size *= output_gradient -> shape[i];
    }
    for(int i = 0; i< axis; i++){
      num_concats *= output_gradient -> shape[i];
    }
    const float * grad_out_data = (const float *)(output_gradient -> data);
    float * grad_in_data = (float *)(input_gradient -> data);
    const int input_concat_axis = input_gradient -> shape[axis];
    const int output_concat_axis = output_gradient -> shape[axis];
  
  
    const int input_concat_size = input_concat_axis * concat_size;
    const int nthreads = input_concat_size * num_concats;
    const int blocks = (nthreads + THREADS_PER_BLOCK - 1)/ THREADS_PER_BLOCK;
    if (stream_handle)
      concat_kernel<<<blocks, THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>(nthreads, grad_out_data, false, num_concats, concat_size,
       output_concat_axis, input_concat_axis, concat_offset, grad_in_data);
    else
      concat_kernel<<<blocks, THREADS_PER_BLOCK>>>(nthreads, grad_out_data, false, num_concats, concat_size,
       output_concat_axis, input_concat_axis, concat_offset, grad_in_data);  
  }
  return 0;
}
