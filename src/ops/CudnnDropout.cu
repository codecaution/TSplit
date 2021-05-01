#include "gpu_runtime.h"
// #include "cuda_device_api.h"

int CuDNN_DLGpuDropout(const DLArrayHandle input_X, const float dropout, DLArrayHandle output_Y,
                       int *reserve_size, void **reserve_space, int first_time ,DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  
  int dev_id = (input_X->ctx).device_id;
  cudnn_init(dev_id, stream_handle);
  size_t input_N,input_C,input_H,input_W;
  // input
  if(input_X->ndim == 2){
    input_N = input_X -> shape[0];
    input_C = input_H = 1;
    input_W = input_X -> shape[1];
  }
  else{
    input_N = input_X -> shape[0];
    input_C = input_X -> shape[1];
    input_H = input_X -> shape[2];
    input_W = input_X -> shape[3];
  }
  const float * input_data = (const float*)(input_X -> data);
  //input descriptor
  cudnnTensorDescriptor_t input_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, 
                                        CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  // dropout descriptor
  cudnnDropoutDescriptor_t dropout_desc;
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));

  unsigned long long seed = 19260817ull; // ha
  size_t state_size;
  CUDNN_CALL(cudnnDropoutGetStatesSize(cudnn_map[dev_id], &state_size));
  // printf("forward:\n");
  // printf("dev_id : %d\n", dev_id);
  // printf("states size: %d\n", states_size);
  void *state_data = MemoryPool -> DLMemoryMalloc(state_size);
  
  // float dropout = 0.5;
  CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc, cudnn_map[dev_id], dropout, state_data, state_size, seed));
  // output
  float *output_data =(float *) output_Y ->data;
  //output descriptor
  cudnnTensorDescriptor_t output_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW,
                                        CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  if(first_time == 1){
    CUDA_CALL(cudaSetDevice(dev_id));
    CUDNN_CALL(cudnnDropoutGetReserveSpaceSize(input_desc, (size_t *)reserve_size));
    // printf("reserve_size: %d\n", *reserve_size);
    // printf("reserve_space pointer: %d\n", *reserve_space);
    *reserve_space = MemoryPool -> DLMemoryMalloc(*((size_t*)reserve_size));
    // printf("reserve_space pointer: %d\n", *reserve_space);
  }
  
  // dropout_forward
  CUDNN_CALL(cudnnDropoutForward(cudnn_map[dev_id], dropout_desc, input_desc, input_data, 
                                 output_desc, output_data, *reserve_space, *reserve_size));

  // CUDA_CALL()
  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));
  // CUDNN_CALL(cudnnDestroy(cudnn));
  if(p != NULL){
    int size_input = 1, size_output = 1;
    for(int i = 0; i < input_X -> ndim; i++)
        size_input *= input_X -> shape[i];
    for(int i = 0; i < output_Y -> ndim; i++)
        size_output *= output_Y -> shape[i];
    p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
  }
  return 0;
}

int CuDNN_DLGpuDropout_gradient(const DLArrayHandle output_Y, const float dropout, DLArrayHandle input_X,
                                int *reserve_size, void **reserve_space, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){

  int dev_id = (input_X->ctx).device_id;
  cudnn_init(dev_id, stream_handle);
  size_t input_N,input_C,input_H,input_W;
  // input
  if(input_X->ndim == 2){
    input_N = input_X -> shape[0];
    input_C = input_H = 1;
    input_W = input_X -> shape[1];
  }
  else{
    input_N = input_X -> shape[0];
    input_C = input_X -> shape[1];
    input_H = input_X -> shape[2];
    input_W = input_X -> shape[3];
  }
  float * input_data = (float*)(input_X -> data);

  //input descriptor
  cudnnTensorDescriptor_t input_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        input_N, input_C, input_H, input_W));

  // dropout descriptor
  cudnnDropoutDescriptor_t dropout_desc;
  CUDNN_CALL(cudnnCreateDropoutDescriptor(&dropout_desc));

  unsigned long long seed = 19260817ull; // ha
  size_t state_size;

  CUDNN_CALL(cudnnDropoutGetStatesSize(cudnn_map[dev_id], &state_size));

  // void *state_data;
  // CUDA_CALL(cudaSetDevice(dev_id));
  // CUDA_CALL(cudaMalloc(&state_data, state_size));
  void *state_data = MemoryPool -> DLMemoryMalloc(state_size);
  CUDNN_CALL(cudnnSetDropoutDescriptor(dropout_desc, cudnn_map[dev_id], dropout, state_data, state_size, seed));

  // output
  const float *output_data = (const float *)(output_Y -> data);

  //output descriptor
  cudnnTensorDescriptor_t output_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                        input_N, input_C, input_H, input_W));

  // dropout_backward
  CUDNN_CALL(cudnnDropoutBackward(cudnn_map[dev_id], dropout_desc, output_desc, output_data, 
                                  input_desc, input_data, *reserve_space, *reserve_size));


  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CALL(cudnnDestroyDropoutDescriptor(dropout_desc));
  if(p != NULL){
    int size_input = 1, size_output = 1;
    for(int i = 0; i < output_Y -> ndim; i++)
        size_input *= output_Y -> shape[i];
    for(int i = 0; i < input_X -> ndim; i++)
        size_output *= input_X -> shape[i];
    p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
  }

  return 0;
}
