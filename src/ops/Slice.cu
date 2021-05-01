#include "gpu_runtime.h"


__global__ void slice_kernel(
    float *out_arr, 
    const float *in_arr, 
    const int64_t *o_shape,
    const int64_t *i_shape,
    const int64_t *begin_pos, 
    size_t ndim, 
    size_t size
) {
    size_t o_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_index >= size) return;

    size_t tmp_index = o_index;
    size_t i_index = 0;
    int64_t i_mat = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int64_t offset = begin_pos[i] + tmp_index % o_shape[i];
        tmp_index /= o_shape[i];
        i_index += offset * i_mat;
        i_mat *= i_shape[i];
    }
    out_arr[o_index] = in_arr[i_index];
}


int DLGpuSlice(
    const DLArrayHandle in_arr, 
    DLArrayHandle out_arr, 
    int64_t *begin_pos, 
    DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL
) {
    assert(in_arr->ndim == out_arr->ndim);
    size_t ndim = in_arr->ndim;
    size_t o_size = 1;
    for (int i = 0; i < ndim; ++i) {
        assert(begin_pos[i] >= 0);
        assert(begin_pos[i] + out_arr->shape[i] <= in_arr->shape[i]);
        o_size *= out_arr ->shape[i];
    }
    const float *i_data = (const float *)in_arr->data;
    float *o_data = (float *)out_arr->data;

    size_t alloc_size = ndim * sizeof(int64_t);
    void *pos = MemoryPool -> DLMemoryMalloc(alloc_size);
    void *i_shape = MemoryPool -> DLMemoryMalloc(alloc_size);
    void *o_shape = MemoryPool -> DLMemoryMalloc(alloc_size);
    CUDA_CALL(cudaMemcpy(pos, (void *)begin_pos, 
        alloc_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(i_shape, (void *)in_arr->shape, 
        alloc_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(o_shape, (void *)out_arr->shape,
        alloc_size, cudaMemcpyHostToDevice));
    dim3 blocks;
    dim3 threads;
    if (o_size <= 1024) {
        threads.x = o_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (o_size + 1023) / 1024;
    }
    if (stream_handle)
        slice_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>
            (o_data, i_data, (const int64_t*)o_shape, (const int64_t*)i_shape, (const int64_t*)pos, ndim, o_size);
    else
        slice_kernel<<<blocks, threads>>>
            (o_data, i_data, (const int64_t*)o_shape, (const int64_t*)i_shape, (const int64_t*)pos, ndim, o_size);
    MemoryPool -> DLMemoryFree(pos, alloc_size);
    MemoryPool -> DLMemoryFree(i_shape, alloc_size);
    MemoryPool -> DLMemoryFree(o_shape, alloc_size);
    if(p != NULL){
      int size_input = 1, size_output = 1;
      for(int i = 0; i < in_arr -> ndim; i++)
          size_input *= in_arr -> shape[i];
      for(int i = 0; i < out_arr -> ndim; i++)
          size_output *= out_arr -> shape[i];
      p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
      p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
      p -> workspace_memory = 0;
    }
    return 0;
}


__global__ void slice_gradient_kernel(
    float *out_arr, 
    const float *in_arr, 
    const int64_t *o_shape,
    const int64_t *i_shape,
    const int64_t *begin_pos, 
    size_t ndim, 
    size_t size
) {
    size_t o_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (o_index >= size) return;

    out_arr[o_index] = 0;

    size_t tmp_index = o_index;
    size_t i_index = 0;
    int64_t i_mat = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        int64_t offset = tmp_index % o_shape[i];
        if (offset < begin_pos[i] || offset >= begin_pos[i] + i_shape[i]) return;
        tmp_index /= o_shape[i];
        i_index += (offset - begin_pos[i]) * i_mat;
        i_mat *= i_shape[i];
    }
    out_arr[o_index] = in_arr[i_index];
}


int DLGpuSliceGradient(
    const DLArrayHandle in_arr, 
    DLArrayHandle out_arr, 
    int64_t *begin_pos, 
    DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL
) {
    assert(in_arr->ndim == out_arr->ndim);
    size_t ndim = in_arr->ndim;
    size_t o_size = 1;
    for (int i = 0; i < ndim; ++i) {
        assert(begin_pos[i] >= 0);
        assert(begin_pos[i] + in_arr->shape[i] <= out_arr->shape[i]);
        o_size *= out_arr ->shape[i];
    }
    const float *i_data = (const float *)in_arr->data;
    float *o_data = (float *)out_arr->data;
    size_t alloc_size = ndim * sizeof(int64_t);
    void *pos = MemoryPool -> DLMemoryMalloc(alloc_size);
    void *i_shape = MemoryPool -> DLMemoryMalloc(alloc_size);
    void *o_shape = MemoryPool -> DLMemoryMalloc(alloc_size);
    CUDA_CALL(cudaMemcpy(pos, (void *)begin_pos, 
        alloc_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(i_shape, (void *)in_arr->shape, 
        alloc_size, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(o_shape, (void *)out_arr->shape,
        alloc_size, cudaMemcpyHostToDevice));
    dim3 blocks;
    dim3 threads;
    if (o_size <= 1024) {
        threads.x = o_size;
        blocks.x = 1;
    } else {
        threads.x = 1024;
        blocks.x = (o_size + 1023) / 1024;
    }
    if (stream_handle)
        slice_gradient_kernel<<<blocks, threads, 0, *(cudaStream_t*)stream_handle->handle>>>
            (o_data, i_data, (const int64_t*)o_shape, (const int64_t*)i_shape, (const int64_t*)pos, ndim, o_size);
    else
        slice_gradient_kernel<<<blocks, threads>>>
            (o_data, i_data, (const int64_t*)o_shape, (const int64_t*)i_shape, (const int64_t*)pos, ndim, o_size);
    MemoryPool -> DLMemoryFree(pos, alloc_size);
    MemoryPool -> DLMemoryFree(i_shape, alloc_size);
    MemoryPool -> DLMemoryFree(o_shape, alloc_size);
    if(p != NULL){
      int size_input = 1, size_output = 1;
      for(int i = 0; i < in_arr -> ndim; i++)
          size_input *= in_arr -> shape[i];
      for(int i = 0; i < out_arr -> ndim; i++)
          size_output *= out_arr -> shape[i];
      p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
      p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
      p -> workspace_memory = 0;
    }
    return 0;
}
