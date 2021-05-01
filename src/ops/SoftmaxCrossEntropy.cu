#include "gpu_runtime.h"
// #include "cuda_device_api.h"
// y = inputs[0], y_ = inputs[1]
// np.mean(-np.sum(y_ * np.log(softmax(y)), axis=1), keepdims=True)
__global__ void matrix_softmax_cross_entropy_kernel(int nrow, int ncol,
                                                    const float *input_a,
                                                    const float *input_b,
                                                    float *output,
                                                    float *loss_per_row) {
  // Two dimensional thread blocks.
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int thread_id = id; thread_id < nrow; thread_id += blockDim.x * gridDim.x)
  {
    float maxval = input_a[thread_id * ncol];
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
      maxval = max(maxval, input_a[thread_id * ncol + x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
      sum += exp(input_a[thread_id * ncol + x] - maxval);
    }
    // Compute per-row loss.
    float loss = 0;
    for (int x = 0; x < ncol; ++x) {
      loss -= input_b[thread_id * ncol + x] * ((input_a[thread_id * ncol + x] - maxval) - log(sum));
    }
    loss_per_row[thread_id] = loss;
  }
  __syncthreads();
  // Compute reduce_mean across rows.
  float mean_loss = 0;
  // Use a single thread to reduce mean across rows.
  if (id == 0) {
    for (int i = 0; i < nrow; ++i) {
      mean_loss += loss_per_row[i];
    }
    mean_loss /= nrow;
    output[0] = mean_loss;
  }
}

int DLGpuSoftmaxCrossEntropy(const DLArrayHandle input_a,
                             const DLArrayHandle input_b,
                             DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  size_t indim = input_a->ndim;
  int nrow = 1;
  for (int i = 0; i < indim-1; ++i) {
    nrow *= input_a->shape[i];
  }
  int ncol = input_a->shape[indim - 1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  float *output_data = (float *)output->data;
  size_t workspace_size = nrow * sizeof(float);
  // int dev_id = (input_a->ctx).device_id;
  // float *work_data = (float*)find_chunk(workspace_size, dev_id);
  void *work_data;
  work_data = MemoryPool -> DLMemoryMalloc(workspace_size);
  // 1 block
  if (stream_handle)
    matrix_softmax_cross_entropy_kernel<<<1, THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>(
      nrow, ncol, input_data_a, input_data_b, output_data, (float*)work_data);
  else
    matrix_softmax_cross_entropy_kernel<<<1, THREADS_PER_BLOCK>>>(
      nrow, ncol, input_data_a, input_data_b, output_data, (float*)work_data);
  // del_chunk(work_data, dev_id);
  MemoryPool -> DLMemoryFree(work_data, workspace_size);
  if(p != NULL){
    int size_input1 = 1, size_input2 = 1, size_output = 1;
    for(int i = 0; i < input_a -> ndim; i++)
      size_input1 *= input_a -> shape[i];
    for(int i = 0; i < input_b -> ndim; i++)
      size_input2 *= input_b -> shape[i];
    for(int i = 0; i < output -> ndim; i++)
      size_output *= output -> shape[i];
    p -> input_memory = 1.0 * (size_input1 + size_input2) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
  }
  return 0;
}

__global__ void softmax_cross_entropy_gradient_kernel(int nrow, int ncol, int batch_size, const float *input_a, const float *input_b, const float *input_c, float *output) {

  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  for (int thread_id = id; thread_id < nrow; thread_id += blockDim.x * gridDim.x)
  {
    float maxval = input_a[thread_id * ncol];
    // Find max for a row.
    for (int x = 1; x < ncol; ++x) {
      maxval = max(maxval, input_a[thread_id * ncol + x]);
    }
    // Deduct by max for a row, and raise to exp.
    float sum = 0;
    for (int x = 0; x < ncol; ++x) {
      sum += exp(input_a[thread_id * ncol + x] - maxval);
    }
    for (int x = 0; x < ncol; ++x) {
      output[thread_id * ncol + x] = (exp(input_a[thread_id * ncol + x] - maxval) / sum - input_b[thread_id * ncol + x]) * input_c[0]/ batch_size;
    }
  }
}

int DLGpuSoftmaxCrossEntropy_Gradient(const DLArrayHandle input_a, const DLArrayHandle input_b,
                                    const DLArrayHandle input_c, DLArrayHandle output,
                                    DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL)
{
  int indim = input_a->ndim;
  int nrow = 1;
  for (int i = 0; i < indim-1; ++i) {
    nrow *= input_a->shape[i];
  }
  int ncol = input_a->shape[indim - 1];
  const float *input_data_a = (const float *)input_a->data;
  const float *input_data_b = (const float *)input_b->data;
  const float *input_data_c = (const float *)input_c ->data;

  const int batch_size = input_a -> shape[0];
  float *output_data = (float *)output->data;

  if (stream_handle)
    softmax_cross_entropy_gradient_kernel<<<1, THREADS_PER_BLOCK, 0, *(cudaStream_t*)stream_handle->handle>>>(
    nrow, ncol, batch_size, input_data_a, input_data_b, input_data_c, output_data);
  else
    softmax_cross_entropy_gradient_kernel<<<1, THREADS_PER_BLOCK>>>(
    nrow, ncol, batch_size, input_data_a, input_data_b, input_data_c, output_data);

  if(p != NULL){
    int size_input1 = 1, size_input2 = 1, size_input3 = 1, size_output = 1;
    for(int i = 0; i < input_a -> ndim; i++)
      size_input1 *= input_a -> shape[i];
    for(int i = 0; i < input_b -> ndim; i++)
      size_input2 *= input_b -> shape[i];
    for(int i = 0; i < input_c -> ndim; i++)
      size_input3 *= input_c -> shape[i];
    for(int i = 0; i < output -> ndim; i++)
      size_output *= output -> shape[i];
    p -> input_memory = 1.0 * (size_input1 + size_input2 + size_input3) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
  }
    return 0;
}
