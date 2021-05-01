#include "gpu_runtime.h"

__global__ void softmax_kernel(int nrow, int ncol, const float *input, float *output) {

    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    for (int thread_id = id; thread_id < nrow; thread_id += blockDim.x * gridDim.x)
    {
      float maxval = input[thread_id * ncol];
      // Find max for a row.
      for (int x = 1; x < ncol; ++x) {
        maxval = max(maxval, input[thread_id * ncol + x]);
      }
      // Deduct by max for a row, and raise to exp.
      float sum = 0;
      for (int x = 0; x < ncol; ++x) {
        sum += exp(input[thread_id * ncol + x] - maxval);
      }
      for (int x = 0; x < ncol; ++x) {
        output[thread_id * ncol + x] = exp(input[thread_id * ncol + x] - maxval) / sum;
      }
    }
  }
  
  int DLGpuSoftmax(const DLArrayHandle input, DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
    /* TODO: Your code here */
    assert (input->ndim == 2);
    assert (output->ndim == 2);
    assert (input->shape[0] == output->shape[0] && input->shape[1] == output->shape[1]);
    int nrow = input->shape[0];
    int ncol = input->shape[1];
    const float *input_data = (const float *)input->data;
    float *output_data = (float *)output->data;
    if (stream_handle)
      softmax_kernel<<<1, THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>(nrow, ncol, input_data, output_data);
    else
      softmax_kernel<<<1, THREADS_PER_BLOCK>>>(nrow, ncol, input_data, output_data);

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
  