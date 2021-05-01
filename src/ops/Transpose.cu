#include "gpu_runtime.h"

__global__ void transpose_kernel(float *odata, const float *idata, const uint *buf, const uint ndims, size_t size) {
  const uint *in_strides = buf;
  const uint *out_strides = buf + ndims;
  const uint *perm = buf + ndims * 2;
  size_t o_idx = blockIdx.x * blockDim.x + threadIdx.x;

  uint i_idx = 0;
  uint t = o_idx;
  for (int i = 0; i < ndims; ++i) {
      const uint ratio = t / out_strides[i];
      t -= ratio * out_strides[i];
      i_idx += ratio * in_strides[perm[i]];
  }
  odata[o_idx] = idata[i_idx];
}


int DLGpuTranspose(const DLArrayHandle input, DLArrayHandle output, int *perm, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  uint ndim = uint(input->ndim);
  uint ndim_ = uint(output->ndim);
  assert (ndim == ndim_);

  int64_t *in_dims = input->shape;
  int64_t *out_dims = output->shape;
  float *input_data = (float *)input->data;
  float *output_data = (float *)output->data;
  
  uint *buf = (uint*)malloc(3 * ndim * sizeof(uint));
  uint *gpu_buf = NULL;

  uint in_stride = 1;
  uint out_stride = 1;
  for (int i = ndim-1; i >= 0; --i) {
      buf[i] = uint(in_stride);
      buf[ndim + i] = uint(out_stride);
      buf[ndim * 2 + i] = uint(perm[i]);
      in_stride *= uint(in_dims[i]);
      out_stride *= uint(out_dims[i]);
  }

  assert (in_stride == out_stride);
  size_t size = in_stride;

  int dev_id = (input->ctx).device_id;
  if(is_chunk_init(dev_id) == false){
      chunk_init(dev_id);
  }
  size_t buf_size = 3 * ndim * sizeof(uint);
//   gpu_buf = (uint *)(MemoryPool -> DLMemoryMalloc(buf_size));
  gpu_buf = (uint*)find_chunk(buf_size, dev_id);
  // std::cout<<"buf size: "<<buf_size<<std::endl;
  // CUDA_CALL(cudaSetDevice(dev_id));
  // CUDA_CALL(cudaMalloc((void**)(&gpu_buf), buf_size));
  CUDA_CALL(cudaMemcpy(gpu_buf, (void*)buf, buf_size, cudaMemcpyHostToDevice));
  // CUDA_CALL(cudaMemcpyAsync(gpu_buf, (void*)buf, buf_size, cudaMemcpyHostToDevice, *(cudaStream_t*)stream_handle->handle));
  dim3 blocks;
  dim3 threads;
  if (size <= 1024) {
      threads.x = size;
      blocks.x = 1;
  } else {
      threads.x = 1024;
      blocks.x = (size + 1023) / 1024;
  }

  if (stream_handle)
      transpose_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(output_data, input_data, gpu_buf, ndim, size);
  else
      transpose_kernel<<<blocks, threads>>>(output_data, input_data, gpu_buf, ndim, size);

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
//   MemoryPool -> DLMemoryFree(gpu_buf, buf_size);
  del_chunk(gpu_buf, dev_id);
  // CUDA_CALL(cudaFree(gpu_buf));
  free(buf);
  return 0;
}

// #include "gpu_runtime.h"

// __global__ void transpose_kernel(float *odata, const float *idata, size_t size) {
//   size_t o_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if(o_idx >= size) return;
//   odata[o_idx] = idata[o_idx];
// }


// int DLGpuTranspose(const DLArrayHandle input, DLArrayHandle output, int *perm, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
//   int size = 1;
//   for(int i = 0; i < input->ndim; i++)
//     size *= (input->shape[i]);
//   float *input_data = (float *)input->data;
//   float *output_data = (float *)output->data;
  
//   dim3 blocks;
//   dim3 threads;
//   if (size <= 1024) {
//       threads.x = size;
//       blocks.x = 1;
//   } else {
//       threads.x = 1024;
//       blocks.x = (size + 1023) / 1024;
//   }
//   if (stream_handle)
//       transpose_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>(output_data, input_data, size);
//   else
//       transpose_kernel<<<blocks, threads>>>(output_data, input_data, size);

//   if(p != NULL){
//     int size_input = 1, size_output = 1;
//     for(int i = 0; i < input -> ndim; i++)
//         size_input *= input -> shape[i];
//     for(int i = 0; i < output -> ndim; i++)
//         size_output *= output -> shape[i];
//     p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
//     p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
//     p -> workspace_memory = 0;
//   }
//   return 0;
// }