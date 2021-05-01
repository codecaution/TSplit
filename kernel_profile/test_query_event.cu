#include<cuda_runtime.h>
#include<cstdio>
#include<iostream>

int main(){
    cudaStream_t cpu2gpu, gpu2cpu;
    cudaStreamCreate(&cpu2gpu);
    cudaStreamCreate(&gpu2cpu);

    cudaEvent_t cpu2gpu_event, gpu2cpu_event;
    cudaEventCreate(&cpu2gpu_event);
    cudaEventCreate(&gpu2cpu_event);

    int size = 1000 * 1000;
    void *dev_ptr;
    void *host_ptr;

    cudaMalloc(&dev_ptr, size);
    cudaMallocHost(&host_ptr, size);

    cudaMemcpyAsync(dev_ptr, host_ptr, size, cudaMemcpyHostToDevice, gpu2cpu);
    cudaEventRecord(gpu2cpu_event, gpu2cpu);

    cudaError_t flags = cudaErrorNotReady;
    while(flags == cudaErrorNotReady){
        flags = cudaEventQuery(gpu2cpu_event);
        if (flags == cudaErrorNotReady){
            std::cout<<"cudaErrorNotReady"<<std::endl;
        }
        else if (flags == cudaSuccess){
            std::cout<<"cudaSuccess"<<std::endl;
        }
        else{
            std::cout<<"Error"<<std::endl;
        }
    }

    return 0;

}