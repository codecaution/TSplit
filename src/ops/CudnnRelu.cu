// #include "gpu_runtime.h"

// // the shape of bn_scale/bias   1*C*1*1
// int CuDNN_DLGpuRelu(const DLArrayHandle input, DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){

//   int dev_id = (input_X->ctx).device_id;
//   cudaSetDevice(dev_id);
//   cudnn_init(dev_id, stream_handle);
//   int input_N = input->shape[0];
//   int input_C = input->shape[1];
//   int input_H = input->shape[2];
//   int input_W = input->shape[3];

//   if(p != NULL){
//     int size_input = 1, size_output = 1;
//     for(int i = 0; i < input -> ndim; i++)
//         size_input *= input -> shape[i];
//     for(int i = 0; i < output -> ndim; i++)
//         size_output *= output -> shape[i];
//     p -> input_memory = 1.0 * (size_input) * sizeof(float) / 1024 / 1024;
//     p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
//     p -> workspace_memory = 0;



//     cudnnTensorDescriptor_t input_desc;
//     CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
//     CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));
    
//     cudnnTensorDescriptor_t output_desc;
//     CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
//     CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));
//     cudnnActivationDescriptor_t activation_desc;
//     CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc));
//     CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));// after conv  

//     float *input_data = (float *)(input->data);
//     float *output_data = (float *)(output->data);
    
//     // Insert the begin and end event.
//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventRecord(start,0);

//     float alpha = 1.0f;
//     float beta = 0.0f;

//     CUDNN_CALL(cudnnActivationForward(cudnn_map[dev_id], activation_desc, &alpha,
//       input_desc, input_data,
//       &beta,
//       output_desc, output_data));    

//     float elapsedTime;
//     cudaEventCreate(&stop);
//     cudaEventRecord(stop,0);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&elapsedTime, start,stop);
//     p->time = elapsedTime; 
//     CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
//     CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
//     CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc));
//     CUDNN_CALL(cudnnDestroy(cudnn_handle));     
//   }else{
//     // input
//     cudnnTensorDescriptor_t input_desc;
//     CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
//     CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));
    
//     cudnnTensorDescriptor_t output_desc;
//     CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
//     CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));
//     cudnnActivationDescriptor_t activation_desc;
//     CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc));
//     CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));// after conv  

//     float *input_data = (float *)(input->data);
//     float *output_data = (float *)(output->data);

//     float alpha = 1.0f;
//     float beta = 0.0f;

//     CUDNN_CALL(cudnnActivationForward(cudnn_handle, activation_desc, &alpha,
//       input_desc, input_data,
//       &beta,
//       output_desc, output_data));    

//     CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
//     CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
//     CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc));
//     CUDNN_CALL(cudnnDestroy(cudnn_handle));     
//   }
//   return 0;
// }

// int CuDNN_DLGpuReluGradient(const DLArrayHandle input, const DLArrayHandle in_grad,
//                              DLArrayHandle output, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL)
//     int dev_id = (input_X->ctx).device_id;
//     cudaSetDevice(dev_id);
//     cudnn_init(dev_id, stream_handle);
//     int input_N = input->shape[0];
//     int input_C = input->shape[1];
//     int input_H = input->shape[2];
//     int input_W = input->shape[3];
//     if(p != NULL){
//       int size_input = 1, size_output = 1;
//       for(int i = 0; i < input -> ndim; i++)
//           size_input *= input -> shape[i];
//       for(int i = 0; i < output -> ndim; i++)
//           size_output *= output -> shape[i];
//       p -> input_memory = 2.0 * (size_input) * sizeof(float) / 1024 / 1024;
//       p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
//       p -> workspace_memory = 0;
  
//       cudnnTensorDescriptor_t input_desc;
//       CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
//       CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));
      
//       cudnnTensorDescriptor_t output_desc;
//       CUDNN_CALL(cudnnCreateTensorDescriptor(&output_desc));
//       CUDNN_CALL(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, input_N, input_C, input_H, input_W));
//       cudnnActivationDescriptor_t activation_desc;
//       CUDNN_CALL(cudnnCreateActivationDescriptor(&activation_desc));
//       CUDNN_CALL(cudnnSetActivationDescriptor(activation_desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0));// after conv  
  
//       float *input_data = (float *)(input->data);
//       float *output_data = (float *)(output->data);
      
//       // Insert the begin and end event.
//       cudaEvent_t start, stop;
//       cudaEventCreate(&start);
//       cudaEventRecord(start,0);
  
//       float alpha = 1.0f;
//       float beta = 0.0f;
  
//       CUDNN_CALL(cudnnActivationForward(cudnn_map[dev_id], activation_desc, &alpha,
//         input_desc, input_data,
//         &beta,
//         output_desc, output_data));    
  
//       float elapsedTime;
//       cudaEventCreate(&stop);
//       cudaEventRecord(stop,0);
//       cudaEventSynchronize(stop);
//       cudaEventElapsedTime(&elapsedTime, start,stop);
//       p->time = elapsedTime; 
//       CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
//       CUDNN_CALL(cudnnDestroyTensorDescriptor(output_desc));
//       CUDNN_CALL(cudnnDestroyActivationDescriptor(activation_desc));
//       CUDNN_CALL(cudnnDestroy(cudnn_handle));       
//     }
    
//   return 0;
// }