#include "gpu_runtime.h"

cublasHandle_t cublas_handle = NULL; 
int DLGpuMatrixMultiply(const DLArrayHandle matA, bool transposeA,
                        const DLArrayHandle matB, bool transposeB,
                        DLArrayHandle matC, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  // TODO: Your code here 
  // Hint: use cublas
  // cublas assume matrix is column major
  assert (matA->ndim == 2);
  assert (matB->ndim == 2);
  assert (matC->ndim == 2);
  
  int dev_id = (matA->ctx).device_id;
  cublas_init(dev_id, stream_handle);

  float one = 1.0f;
  float zero = 0.0f;
  int m = matC->shape[1];
  int n = matC->shape[0];
  int k = transposeA ? matA->shape[0] : matA->shape[1];

  if(p != NULL){
    int size_a = 1, size_b = 1, size_c = 1;
    for(int i = 0; i < matA -> ndim; i++)
        size_a *= matA -> shape[i];
    for(int i = 0; i < matB -> ndim; i++)
        size_b *= matB -> shape[i];
    for(int i = 0; i < matC -> ndim; i++)
        size_c *= matC -> shape[i];
    p -> input_memory = 1.0 * (size_a  + size_b) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_c * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    cublasSgemm(cublas_map[dev_id], 
      transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
      m, n, k,
      &one,
      (const float *) matB->data, !transposeB ? m : k,
      (const float *) matA->data, !transposeA ? k : n,
      &zero,
      (float *) matC->data, m
    );
    
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
    cublasSgemm(cublas_map[dev_id], 
      transposeB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transposeA ? CUBLAS_OP_T : CUBLAS_OP_N,
      m, n, k,
      &one,
      (const float *) matB->data, !transposeB ? m : k,
      (const float *) matA->data, !transposeA ? k : n,
      &zero,
      (float *) matC->data, m
    );
  }
  return 0;
}
