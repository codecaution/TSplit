#include "gpu_runtime.h"
#include <bits/stdc++.h>
using namespace std;
// #include "cuda_device_api.h"

int CuDNN_DLGpuConv2d(const DLArrayHandle input_x, const DLArrayHandle input_f, DLArrayHandle output,const int padding1, const int padding2, const int stride, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
    // printf("CuDNN_DLGpuConv2d\n");
    if(p != NULL){ // This is in profiling phase.
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventRecord(start,0);
        int size_input_x = 1, size_input_f = 1, size_output = 1;
        for(int i = 0; i < input_x -> ndim; i++)
            size_input_x *= input_x -> shape[i];
        for(int i = 0; i < input_f -> ndim; i++)
            size_input_f *= input_f -> shape[i];
        for(int i = 0; i < output -> ndim; i++)
            size_output *= output -> shape[i];
        p -> input_memory = 1.0 * (size_input_x + size_input_f) * sizeof(float) / 1024 / 1024;
        p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;

        // Insert the begin and end event.

        int dev_id = (input_x->ctx).device_id;
        cudnn_init(dev_id, stream_handle);
        
        size_t input_N = input_x->shape[0];
        size_t input_C = input_x->shape[1];
        size_t input_H = input_x->shape[2];
        size_t input_W = input_x->shape[3];
        const float *input_data = (const float*) input_x->data;
        
        // input
        cudnnTensorDescriptor_t input_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,input_N,input_C,input_H,input_W));
        size_t filter_N = input_f->shape[0];
        size_t filter_C = input_f->shape[1];
        size_t filter_H = input_f->shape[2];
        size_t filter_W = input_f->shape[3];
        const float* filter_data = (const float*)input_f ->data;
       
        //filter
        cudnnFilterDescriptor_t filter_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,filter_N,filter_C,filter_H,filter_W));
    
        //convolution
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
              conv_desc,
              padding1, padding2, stride, stride, 1, 1,
              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        size_t out_N = output->shape[0];
        size_t out_C = output->shape[1];
        size_t out_H = output->shape[2];
        size_t out_W = output->shape[3];
        //output
        cudnnTensorDescriptor_t out_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,out_N, out_C, out_H, out_W));
        float *output_data = (float *)output -> data;
        //algorithm
        cudnnConvolutionFwdAlgo_t algo;
        CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn_map[dev_id],input_desc,filter_desc, conv_desc,out_desc,CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,0,&algo));

        // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn_map[dev_id],input_desc,filter_desc, conv_desc,out_desc,CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,&algo));
        // algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        // algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        // algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
        // algo = CUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
        // algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT;
        // algo = CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING;
        // algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD;
        // algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
        size_t workspace_size;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          cudnn_map[dev_id], input_desc, filter_desc, conv_desc, out_desc, algo, &workspace_size));

        // if(algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM)
        //   std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
        // else if(algo == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
        //   std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
        // else if(algo == CUDNN_CONVOLUTION_FWD_ALGO_GEMM)
        //   std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_GEMM:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
        // else if(algo == CUDNN_CONVOLUTION_FWD_ALGO_DIRECT)
        //   std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_DIRECT:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
        // else if(algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT)
        //   std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_FFT:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
        // else if(algo == CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING)
        //   std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
        // else if(algo == CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD)
        //   std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
        // else
        //   std::cout<<"CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
        // std::cout<<"memory pool conv forward workspace size= "<< workspace_size/1024/1024<<std::endl;
        void *work_data;
        if(workspace_size != 0)
          work_data = MemoryPool -> DLMemoryMalloc(workspace_size);
        // std::cout<<"forward  "<<std::endl;
        float alpha = 1.0f;
        float beta = 0.0f;
        CUDNN_CALL(cudnnConvolutionForward(
          cudnn_map[dev_id],
          &alpha, input_desc, input_data, filter_desc, filter_data,
          conv_desc, algo, work_data, workspace_size,
          &beta, out_desc, output_data));
        // std::cout<<"end  "<<std::endl;
        if(workspace_size != 0){
          MemoryPool -> DLMemoryFree(work_data, workspace_size);
        }
        CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
        CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
        CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));

        float elapsedTime;
        cudaEventCreate(&stop);
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start,stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        p -> workspace_memory = workspace_size / 1024 / 1024;
        p->time = elapsedTime;
      }else{
        int dev_id = (input_x->ctx).device_id;
        cudnn_init(dev_id, stream_handle);
        size_t input_N = input_x->shape[0];
        size_t input_C = input_x->shape[1];
        size_t input_H = input_x->shape[2];
        size_t input_W = input_x->shape[3];
        const float *input_data = (const float*) input_x->data;
        
        // input
        cudnnTensorDescriptor_t input_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,input_N,input_C,input_H,input_W));
        size_t filter_N = input_f->shape[0];
        size_t filter_C = input_f->shape[1];
        size_t filter_H = input_f->shape[2];
        size_t filter_W = input_f->shape[3];
        const float* filter_data = (const float*)input_f ->data;
       
        //filter
        cudnnFilterDescriptor_t filter_desc;
        CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
        CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,filter_N,filter_C,filter_H,filter_W));
    
        //convolution
        cudnnConvolutionDescriptor_t conv_desc;
        CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
        CUDNN_CALL(cudnnSetConvolution2dDescriptor(
              conv_desc,
              padding1, padding2, stride, stride, 1, 1,
              CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        size_t out_N = output->shape[0];
        size_t out_C = output->shape[1];
        size_t out_H = output->shape[2];
        size_t out_W = output->shape[3];
        // std::cout<<input_N<<" "<<input_C<<" "<<input_H<<" "<<input_W<<"\n";
        // std::cout<<filter_N<<" "<<filter_C<<" "<<filter_H<<" "<<filter_W<<"\n";
        // std::cout<<out_N<<" "<<out_C<<" "<<out_H<<" "<<out_W<<"\n";
        // std::cout<<padding1<<" "<<padding2<<" "<<stride<<"\n";
        //output
        cudnnTensorDescriptor_t out_desc;
        CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
        CUDNN_CALL(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,out_N, out_C, out_H, out_W));
        float *output_data = (float *)output -> data;
        //algorithm
        cudnnConvolutionFwdAlgo_t algo;
        CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn_map[dev_id],input_desc,filter_desc, conv_desc,out_desc,CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,0,&algo));
        // CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(cudnn_map[dev_id],input_desc,filter_desc, conv_desc,out_desc,CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,0,&algo));
    
        size_t workspace_size;
        CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
          cudnn_map[dev_id], input_desc, filter_desc, conv_desc, out_desc, algo, &workspace_size));

        // void *work_data = find_chunk(workspace_size, dev_id);
        void *work_data = NULL;
//        cout << 123 << endl;
//        cout << MemoryPool << endl;
        if(workspace_size != 0)
          work_data = MemoryPool -> DLMemoryMalloc(workspace_size);
        // std::cout<<"workspace_size"<<workspace_size<<std::endl;
//        cout << 456 << endl;
        float alpha = 1.0f;
        float beta = 0.0f;
        CUDNN_CALL(cudnnConvolutionForward(
          cudnn_map[dev_id],
          &alpha, input_desc, input_data, filter_desc, filter_data,
          conv_desc, algo, work_data, workspace_size,
          &beta, out_desc, output_data));
        // del_chunk(work_data, dev_id);
//        cout << 789 << endl;
        if(workspace_size != 0)
          MemoryPool -> DLMemoryFree(work_data, workspace_size);
//        cout << 101112 << endl;
        CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
        CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
        CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
        CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));    
      }
    return 0;
}
int CuDNN_DLGpuConv2d_Gradient_of_Filter(const DLArrayHandle input_x, const DLArrayHandle gradient_y, DLArrayHandle gradient_f, const int padding1, const int padding2, const int stride, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  // printf("CuDNN_DLGpuConv2d_Gradient_of_Filter\n"); 
  if(p != NULL){
    int size_input_x = 1, size_grad_y = 1, size_output = 1;
    for(int i = 0; i < input_x -> ndim; i++)
        size_input_x *= input_x -> shape[i];
    for(int i = 0; i < gradient_y -> ndim; i++)
        size_grad_y *= gradient_y -> shape[i];
    for(int i = 0; i < gradient_f -> ndim; i++)
        size_output *= gradient_f -> shape[i];
    p -> input_memory = 1.0 * (size_input_x + size_grad_y) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
    
    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    int dev_id = (input_x->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    // input
    size_t input_N = input_x->shape[0];
    size_t input_C = input_x->shape[1];
    size_t input_H = input_x->shape[2];
    size_t input_W = input_x->shape[3];
    const float *input_data = (const float*) input_x->data;
    
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,input_N,input_C,input_H,input_W));
    // dy
    size_t dy_N = gradient_y -> shape[0];
    size_t dy_C = gradient_y -> shape[1];
    size_t dy_H = gradient_y -> shape[2];
    size_t dy_W = gradient_y -> shape[3];
    const float *dy_data = (const float*)gradient_y ->data;
    
    cudnnTensorDescriptor_t dy_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dy_N, dy_C, dy_H, dy_W));

    //conv2d
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          conv_desc,
          padding1, padding2, stride, stride, 1, 1,
          CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    //dw 
    size_t df_N = gradient_f->shape[0];
    size_t df_C = gradient_f->shape[1];
    size_t df_H = gradient_f->shape[2];
    size_t df_W = gradient_f->shape[3];
    float *df_data = (float*) gradient_f->data;

    cudnnFilterDescriptor_t df_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&df_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(df_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, df_N, df_C, df_H, df_W));
    
    //algo
    cudnnConvolutionBwdFilterAlgo_t algo;
    
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,0, &algo));
    // CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc,
    //                                                       CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,0, &algo));
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3;
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING;
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;

    size_t workspace_size;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc, algo, &workspace_size));
    // std::cout<<"require workspace = "<<workspace_size<<std::endl;
    // if(algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // std::cout<<"memory pool conv backward filter workspace size= "<< workspace_size/1024/1024<<std::endl;
    void *work_data;
    if(workspace_size != 0)
      work_data = MemoryPool -> DLMemoryMalloc(workspace_size);

    // printf("start to compute...\n");
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardFilter(
      cudnn_map[dev_id],
      &alpha, input_desc, input_data, dy_desc, dy_data,
      conv_desc, algo, work_data, workspace_size,
      &beta, df_desc, df_data));
    // printf("compute end ...\n");
    // del_chunk(work_data, dev_id);
    if(workspace_size != 0)
      MemoryPool -> DLMemoryFree(work_data, workspace_size);

    CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(df_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
    
    // printf("event sync ...\n");
    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    // printf("event sync 1 ...\n");
    cudaEventSynchronize(stop);
    // printf("event sync 2 ...\n");
    cudaEventElapsedTime(&elapsedTime, start,stop);
    // printf("event sync end ...\n");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    p->time = elapsedTime;
    p -> workspace_memory = workspace_size / 1024 / 1024;
  }else{
    int dev_id = (input_x->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
    // input
    size_t input_N = input_x->shape[0];
    size_t input_C = input_x->shape[1];
    size_t input_H = input_x->shape[2];
    size_t input_W = input_x->shape[3];
    const float *input_data = (const float*) input_x->data;
    
    cudnnTensorDescriptor_t input_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,input_N,input_C,input_H,input_W));
    // dy
    size_t dy_N = gradient_y -> shape[0];
    size_t dy_C = gradient_y -> shape[1];
    size_t dy_H = gradient_y -> shape[2];
    size_t dy_W = gradient_y -> shape[3];
    const float *dy_data = (const float*)gradient_y ->data;
    
    cudnnTensorDescriptor_t dy_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dy_N, dy_C, dy_H, dy_W));

    //conv2d
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          conv_desc,
          padding1, padding2, stride, stride, 1, 1,
          CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    //dw 
    size_t df_N = gradient_f->shape[0];
    size_t df_C = gradient_f->shape[1];
    size_t df_H = gradient_f->shape[2];
    size_t df_W = gradient_f->shape[3];
    float *df_data = (float*) gradient_f->data;

    cudnnFilterDescriptor_t df_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&df_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(df_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, df_N, df_C, df_H, df_W));
    
    //algo
    cudnnConvolutionBwdFilterAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc,
      CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE,0, &algo));
    // CUDNN_CALL(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc,
    //                                                       CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,0, &algo));
    // std::cout<<"conv2d backward filter: "<<algo<<std::endl;
    //algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT;
    // algo = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
    size_t workspace_size;
    CUDNN_CALL(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_map[dev_id], input_desc, dy_desc, conv_desc, df_desc, algo, &workspace_size));
    // void *work_data = find_chunk(workspace_size, dev_id);
    void *work_data = NULL;
    if(workspace_size != 0)
      work_data = MemoryPool -> DLMemoryMalloc(workspace_size);
    // std::cout<<"workspace_size"<<workspace_size<<std::endl;
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardFilter(
      cudnn_map[dev_id],
      &alpha, input_desc, input_data, dy_desc, dy_data,
      conv_desc, algo, work_data, workspace_size,
      &beta, df_desc, df_data));    
    // del_chunk(work_data, dev_id);
    if(workspace_size != 0)
        MemoryPool -> DLMemoryFree(work_data, workspace_size);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(df_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_desc));
  }
  return 0;
}

  
int CuDNN_DLGpuConv2d_Gradient_of_Data(const DLArrayHandle input_f, const DLArrayHandle gradient_y, DLArrayHandle gradient_x,const int padding1, const int padding2, const int stride, DLStreamHandle stream_handle = NULL, ProfilerHandle p = NULL){
  // printf("CuDNN_DLGpuConv2d_Gradient_of_Data\n");
  if(p != NULL){
    int size_input_f = 1, size_grad_y = 1, size_output = 1;
    for(int i = 0; i < input_f -> ndim; i++)
        size_input_f *= input_f -> shape[i];
    for(int i = 0; i < gradient_y -> ndim; i++)
        size_grad_y *= gradient_y -> shape[i];
    for(int i = 0; i < gradient_x -> ndim; i++)
        size_output *= gradient_x -> shape[i];
    p -> input_memory = 1.0 * (size_input_f + size_grad_y) * sizeof(float) / 1024 / 1024;
    p -> output_memory = 1.0 * size_output * sizeof(float) / 1024 / 1024;
    // Insert the begin and end event.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventRecord(start,0);

    int dev_id = (input_f->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
      
    // filter
    size_t filter_N = input_f->shape[0];
    size_t filter_C = input_f->shape[1];
    size_t filter_H = input_f->shape[2];
    size_t filter_W = input_f->shape[3];
    const float *filter_data = (const float*) input_f->data;
    
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW, filter_N, filter_C, filter_H, filter_W));
    // dy
    size_t dy_N = gradient_y -> shape[0];
    size_t dy_C = gradient_y -> shape[1];
    size_t dy_H = gradient_y -> shape[2];
    size_t dy_W = gradient_y -> shape[3];
    const float *dy_data = (const float*)gradient_y ->data;
    
    cudnnTensorDescriptor_t dy_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dy_N, dy_C, dy_H, dy_W));
  
    //conv2d
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          conv_desc,
          padding1, padding2, stride, stride, 1, 1,
          CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    //dx 
    size_t dx_N = gradient_x->shape[0];
    size_t dx_C = gradient_x->shape[1];
    size_t dx_H = gradient_x->shape[2];
    size_t dx_W = gradient_x->shape[3];
    float *dx_data = (float*) gradient_x->data;
  
    cudnnTensorDescriptor_t dx_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dx_N, dx_C, dx_H, dx_W));
    
    //algo
    cudnnConvolutionBwdDataAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc,
                                                        CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,0, &algo));
    // CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc,
    //                                                     CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,0, &algo));
  
    // algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
    // algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    // algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT;
    // algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING;
    // algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD;
    // algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;

    size_t workspace_size;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc, algo, &workspace_size));

    // if(algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_0)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_DATA_ALGO_0:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_1)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_DATA_ALGO_1:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // else if(algo == CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED)
    //   std::cout<<"CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:"<<algo<<" workspace size = "<<workspace_size/(1024*1024)<<" MB"<<std::endl;
    // std::cout<<"memory pool conv backward data workspace size= "<< workspace_size/1024/1024<<std::endl;
    void *work_data;
    if(workspace_size != 0)
      work_data = MemoryPool -> DLMemoryMalloc(workspace_size);
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardData(
      cudnn_map[dev_id],
      &alpha, filter_desc, filter_data, dy_desc, dy_data,
      conv_desc, algo, work_data, workspace_size,
      &beta, dx_desc, dx_data));
    if(workspace_size != 0)
      MemoryPool -> DLMemoryFree(work_data, workspace_size);
    // del_chunk(work_data, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));
  
    float elapsedTime;
    cudaEventCreate(&stop);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start,stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    p->time = elapsedTime; 
    p -> workspace_memory = workspace_size / 1024 / 1024;
  }else{
    int dev_id = (input_f->ctx).device_id;
    cudnn_init(dev_id, stream_handle);
      
    // filter
    size_t filter_N = input_f->shape[0];
    size_t filter_C = input_f->shape[1];
    size_t filter_H = input_f->shape[2];
    size_t filter_W = input_f->shape[3];
    const float *filter_data = (const float*) input_f->data;
    
    cudnnFilterDescriptor_t filter_desc;
    CUDNN_CALL(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CALL(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW, filter_N, filter_C, filter_H, filter_W));
    // dy
    size_t dy_N = gradient_y -> shape[0];
    size_t dy_C = gradient_y -> shape[1];
    size_t dy_H = gradient_y -> shape[2];
    size_t dy_W = gradient_y -> shape[3];
    const float *dy_data = (const float*)gradient_y ->data;
    
    cudnnTensorDescriptor_t dy_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dy_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(dy_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dy_N, dy_C, dy_H, dy_W));
  
    //conv2d
    cudnnConvolutionDescriptor_t conv_desc;
    CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
    CUDNN_CALL(cudnnSetConvolution2dDescriptor(
          conv_desc,
          padding1, padding2, stride, stride, 1, 1,
          CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    //dx 
    size_t dx_N = gradient_x->shape[0];
    size_t dx_C = gradient_x->shape[1];
    size_t dx_H = gradient_x->shape[2];
    size_t dx_W = gradient_x->shape[3];
    float *dx_data = (float*) gradient_x->data;
  
    cudnnTensorDescriptor_t dx_desc;
    CUDNN_CALL(cudnnCreateTensorDescriptor(&dx_desc));
    CUDNN_CALL(cudnnSetTensor4dDescriptor(dx_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, dx_N, dx_C, dx_H, dx_W));
    
    //algo
    cudnnConvolutionBwdDataAlgo_t algo;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc,
      CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE,0, &algo));
    // CUDNN_CALL(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc,
    //                                                     CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,0, &algo));
  
    size_t workspace_size;
    CUDNN_CALL(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_map[dev_id], filter_desc, dy_desc, conv_desc, dx_desc, algo, &workspace_size));
    // void *work_data = find_chunk(workspace_size, dev_id);  
    
    void *work_data = NULL;
    if(workspace_size != 0)
      work_data = MemoryPool -> DLMemoryMalloc( workspace_size);
    // std::cout<<"workspace_size"<<workspace_size<<std::endl;
    float alpha = 1.0f;
    float beta = 0.0f;
    CUDNN_CALL(cudnnConvolutionBackwardData(
      cudnn_map[dev_id],
      &alpha, filter_desc, filter_data, dy_desc, dy_data,
      conv_desc, algo, work_data, workspace_size,
      &beta, dx_desc, dx_data));
    if(workspace_size != 0)
      MemoryPool -> DLMemoryFree(work_data, workspace_size);
    // del_chunk(work_data, dev_id);
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dy_desc));
    CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(dx_desc));
    CUDNN_CALL(cudnnDestroyFilterDescriptor(filter_desc));  
  }
  return 0;
}
