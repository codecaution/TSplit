#include "gpu_runtime.h"
// #include "cuda_device_api.h"


__global__ void init_mean_and_var_kernel(float *mean, float *var, size_t size){
  size_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= size) return;
  mean[id] = 0;
  var[id] = 0;
}
// the shape of bn_scale/bias   1*C*1*1
int CuDNN_DLGpuBatch_Normalization(const DLArrayHandle input_X, const DLArrayHandle bn_scale, const DLArrayHandle bn_bias, DLArrayHandle output_Y,
                                   float momentum , float eps, DLArrayHandle save_mean_arr = NULL, DLArrayHandle save_var_arr = NULL, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){

  int dev_id = (input_X->ctx).device_id;
  cudaSetDevice(dev_id);
  cudnn_init(dev_id, stream_handle);
  if(p != NULL){
    int size_a = 1, size_b = 1, size_c = 1, size_d = 1, size_e = 1, size_f = 1;
    for(int i = 0; i < input_X -> ndim; i++)
        size_a *= input_X -> shape[i];
    for(int i = 0; i < bn_scale -> ndim; i++)
        size_b *= bn_scale -> shape[i];
    for(int i = 0; i < bn_bias -> ndim; i++)
        size_c *= bn_bias -> shape[i];
    for(int i = 0; i < output_Y -> ndim; i++)
        size_d *= output_Y -> shape[i];
    if(save_mean_arr != NULL){
      for(int i = 0; i < save_mean_arr -> ndim; i++)
          size_e *= save_mean_arr -> shape[i];
    }
    if(save_var_arr != NULL){
      for(int i = 0; i < save_var_arr -> ndim; i++)
          size_f *= save_var_arr -> shape[i];
    }
    p -> input_memory = 1.0 * (size_a  + size_b + size_c) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * (size_d + size_e + size_f) * sizeof(float) / 1024 / 1024;
    
    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    // input
    size_t input_N = input_X -> shape[0];
    size_t input_C = input_X -> shape[1];
    size_t input_H = input_X -> shape[2];
    size_t input_W = input_X -> shape[3];
    const float * input_data = (const float*)(input_X -> data);
    
    //input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

    //output 
    float *output_data = (float *)(output_Y -> data);

    //output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

    //bn parameter descriptor
    cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
    CUDNN_CALL(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));// after conv

    //bn parameter
    const float *bn_scale_data = (const float *) (bn_scale -> data);
    const float *bn_bias_data = (const float *) (bn_bias -> data);

    size_t workspace_size = sizeof(float) * input_C;
    // void* running_mean = find_chunk(workspace_size, dev_id);
    // void* running_var = find_chunk(workspace_size, dev_id);
    void *running_mean, *running_var;
    running_mean = MemoryPool -> DLMemoryMalloc( sizeof(float) * input_C);
    running_var = MemoryPool -> DLMemoryMalloc( sizeof(float) * input_C);
    // init running mean and running var
    // the tensors should be initialized to some reasonable values or to 0.
    size_t blocks = (input_C + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if(stream_handle){
      cudaStream_t *s = (cudaStream_t*)(stream_handle->handle);
      init_mean_and_var_kernel<<<blocks, THREADS_PER_BLOCK, 0, *s>>>((float*)running_mean, (float*)running_var, input_C);
    }
    else{
      init_mean_and_var_kernel<<<blocks, THREADS_PER_BLOCK>>>((float*)running_mean, (float*)running_var, input_C);
    }
    /************************************************************/
    void *save_mean;
    void *save_var;
    
    if(save_mean_arr != NULL)
      save_mean = save_mean_arr -> data;
    else
      save_mean = NULL;

    if(save_var_arr != NULL)
      save_var = save_var_arr -> data;
    else
      save_var = NULL;

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta,
                                                      input_desc, input_data,
                                                      output_desc, output_data,
                                                      bnScaleBiasMeanVar_desc,
                                                      bn_scale_data, bn_bias_data,
                                                      momentum, running_mean, running_var,
                                                      eps, save_mean, save_var));
    // del_chunk(running_mean, dev_id);
    // del_chunk(running_var, dev_id);

    MemoryPool -> DLMemoryFree(running_mean, sizeof(float) * input_C);
    MemoryPool -> DLMemoryFree(running_var, sizeof(float) * input_C);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));

    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    p->time = elapsedTime;
    p -> workspace_memory = (workspace_size + workspace_size)/ 1024 / 1024;
  }else{
    // input
    size_t input_N = input_X -> shape[0];
    size_t input_C = input_X -> shape[1];
    size_t input_H = input_X -> shape[2];
    size_t input_W = input_X -> shape[3];
    const float * input_data = (const float*)(input_X -> data);
    
    //input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

    //output 
    float *output_data = (float *)(output_Y -> data);

    //output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

    //bn parameter descriptor
    cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
    CUDNN_CALL(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));// after conv

    //bn parameter
    const float *bn_scale_data = (const float *) (bn_scale -> data);
    const float *bn_bias_data = (const float *) (bn_bias -> data);

    // size_t workspace_size = sizeof(float) * input_C;
    // void* running_mean = find_chunk(workspace_size, dev_id);
    // void* running_var = find_chunk(workspace_size, dev_id);
    void *running_mean, *running_var;
    running_mean = MemoryPool -> DLMemoryMalloc( sizeof(float) * input_C);
    running_var = MemoryPool -> DLMemoryMalloc( sizeof(float) * input_C);
    // init running mean and running var
    // the tensors should be initialized to some reasonable values or to 0.
    size_t blocks = (input_C + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if(stream_handle){
      cudaStream_t *s = (cudaStream_t*)(stream_handle->handle);
      init_mean_and_var_kernel<<<blocks, THREADS_PER_BLOCK, 0, *s>>>((float*)running_mean, (float*)running_var, input_C);
    }
    else{
      init_mean_and_var_kernel<<<blocks, THREADS_PER_BLOCK>>>((float*)running_mean, (float*)running_var, input_C);
    }
    /************************************************************/
    void *save_mean;
    void *save_var;
    
    if(save_mean_arr != NULL)
      save_mean = save_mean_arr -> data;
    else
      save_mean = NULL;

    if(save_var_arr != NULL)
      save_var = save_var_arr -> data;
    else
      save_var = NULL;

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnBatchNormalizationForwardTraining(cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta,
                                                      input_desc, input_data,
                                                      output_desc, output_data,
                                                      bnScaleBiasMeanVar_desc,
                                                      bn_scale_data, bn_bias_data,
                                                      momentum, running_mean, running_var,
                                                      eps, save_mean, save_var));
    // del_chunk(running_mean, dev_id);
    // del_chunk(running_var, dev_id);                
    MemoryPool -> DLMemoryFree(running_mean, sizeof(float) * input_C);
    MemoryPool -> DLMemoryFree(running_var, sizeof(float) * input_C);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));
  }
  return 0;
}

int CuDNN_DLGpuBatch_Normalization_gradient(const DLArrayHandle gradient_Y, const DLArrayHandle input_X, const DLArrayHandle bn_scale, DLArrayHandle gradient_X, 
                                            DLArrayHandle gradient_bn_scale, DLArrayHandle gradient_bn_bias, float eps, DLArrayHandle save_mean_arr = NULL,
                                            DLArrayHandle save_var_arr = NULL, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  // CUDNN_CALL(cudnnCreate(&cudnn));
  int dev_id = (input_X->ctx).device_id;
  cudnn_init(dev_id, stream_handle);

  // CUDNN_CALL(cudnnDestroy(cudnn));
  if(p != NULL){
    int size_a = 1, size_b = 1, size_c = 1, size_d = 1, size_e = 1, size_f = 1, size_g = 1, size_h = 1;
    for(int i = 0; i < gradient_Y -> ndim; i++)
        size_a *= gradient_Y -> shape[i];
    for(int i = 0; i < input_X -> ndim; i++)
        size_b *= input_X -> shape[i];
    for(int i = 0; i < bn_scale -> ndim; i++)
        size_c *= bn_scale -> shape[i];
    for(int i = 0; i < gradient_X -> ndim; i++)
        size_d *= gradient_X -> shape[i];
    for(int i = 0; i < gradient_bn_scale -> ndim; i++)
        size_e *= gradient_bn_scale -> shape[i];
    for(int i = 0; i < gradient_bn_bias -> ndim; i++)
        size_f *= gradient_bn_bias -> shape[i];
    if(save_mean_arr != NULL){
      for(int i = 0; i < save_mean_arr -> ndim; i++)
          size_g *= save_mean_arr -> shape[i];
    }
    if(save_var_arr != NULL){
      for(int i = 0; i < save_var_arr -> ndim; i++)
          size_h *= save_var_arr -> shape[i];
    }
    p -> input_memory = 1.0 * (size_a  + size_b + size_c + size_g + size_h) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * (size_d + size_e + size_f) * sizeof(float) / 1024 / 1024;
    p -> workspace_memory = 0;
    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    // input
    size_t input_N = input_X -> shape[0];
    size_t input_C = input_X -> shape[1];
    size_t input_H = input_X -> shape[2];
    size_t input_W = input_X -> shape[3];
    const float * input_data = (const float*)(input_X -> data);
    
    //input descriptor
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

    //output 
    const float *gradient_y_data = (const float *)(gradient_Y -> data);

    //output descriptor
    cudnnTensorDescriptor_t output_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

    //bn parameter descriptor
    cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
    CUDNN_CALL(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));// after conv
    
    const float * bn_scale_data = (const float *)(bn_scale->data);

    // x gradient
    float *gradient_x_data = (float *) (gradient_X -> data);
    //bn gradient
    float *gradient_bn_bias_data = (float *)(gradient_bn_bias -> data);
    float *gradient_bn_scale_data = (float *)(gradient_bn_scale -> data);
    void *saved_mean;
    void *saved_var;
    
    if(save_mean_arr != NULL)
      saved_mean = save_mean_arr -> data;
    else
      saved_mean = NULL;

    if(save_var_arr != NULL)
      saved_var = save_var_arr -> data;
    else
      saved_var = NULL;
    float one = 1.0f;
    float zero = 0.0f;
    // std::cout<<input_data<<" "<<gradient_y_data<<" "<<gradient_x_data<<" "<<gradient_bn_scale_data<<" "<<gradient_bn_bias_data<<std::endl;
    CUDNN_CALL(cudnnBatchNormalizationBackward(cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &one, &zero, &one, &zero,
                                              input_desc, input_data,
                                              output_desc, gradient_y_data,
                                              input_desc, gradient_x_data,
                                              bnScaleBiasMeanVar_desc,
                                              bn_scale_data,
                                              gradient_bn_scale_data,
                                              gradient_bn_bias_data,
                                              eps, saved_mean, saved_var));
                                              
                                                
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));  
    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    p->time = elapsedTime;
  }else{
      // input
  size_t input_N = input_X -> shape[0];
  size_t input_C = input_X -> shape[1];
  size_t input_H = input_X -> shape[2];
  size_t input_W = input_X -> shape[3];
  const float * input_data = (const float*)(input_X -> data);
  
  //input descriptor
  cudnnTensorDescriptor_t input_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  //output 
  const float *gradient_y_data = (const float *)(gradient_Y -> data);

  //output descriptor
  cudnnTensorDescriptor_t output_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));

  //bn parameter descriptor
  cudnnTensorDescriptor_t bnScaleBiasMeanVar_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&bnScaleBiasMeanVar_desc));
  CUDNN_CALL(cudnnDeriveBNTensorDescriptor(bnScaleBiasMeanVar_desc, input_desc, CUDNN_BATCHNORM_SPATIAL));// after conv
  
  const float * bn_scale_data = (const float *)(bn_scale->data);

  // x gradient
  float *gradient_x_data = (float *) (gradient_X -> data);
  //bn gradient
  float *gradient_bn_bias_data = (float *)(gradient_bn_bias -> data);
  float *gradient_bn_scale_data = (float *)(gradient_bn_scale -> data);
  void *saved_mean;
  void *saved_var;
  
  if(save_mean_arr != NULL)
    saved_mean = save_mean_arr -> data;
  else
    saved_mean = NULL;

  if(save_var_arr != NULL)
    saved_var = save_var_arr -> data;
  else
    saved_var = NULL;
  float one = 1.0f;
  float zero = 0.0f;

  CUDNN_CALL(cudnnBatchNormalizationBackward(cudnn_map[dev_id], CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &one, &zero, &one, &zero,
                                             input_desc, input_data,
                                             output_desc, gradient_y_data,
                                             input_desc, gradient_x_data,
                                             bnScaleBiasMeanVar_desc,
                                             bn_scale_data,
                                             gradient_bn_scale_data,
                                             gradient_bn_bias_data,
                                             eps, saved_mean, saved_var));
                                            
                                              
  CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(bnScaleBiasMeanVar_desc));
  }
  return 0;
}