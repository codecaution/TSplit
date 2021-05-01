#include "../header/nccl_communication.h"
__global__ void array_set(float *a, float tmp, int size){
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if(id >= size) return;
  a[id] = tmp;
}

void create_streams(cudaStream_t *stream, int *devices, int devices_numbers){
  // stream = (cudaStream_t *) stream;
  // *stream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * devices_numbers);
  for(int i = 0; i < devices_numbers; i++){
		CUDACHECK(cudaSetDevice(devices[i]));
		CUDACHECK(cudaStreamCreate((cudaStream_t *)stream + i));
  }
}

/*
void update_stream(size_t dev_id, cudaStream_t *stream, DLStreamHandle stream_handle){
  stream[dev_id] = *(cudaStream_t*)stream_handle->handle;
}
*/
void update_stream(size_t dev_id, cudaStream_t *stream, cudaStream_t *stream_handle){
  stream[dev_id] = *stream_handle;
}

void free_streams(cudaStream_t *stream, int *devices, int devices_numbers){
	for(int i = 0; i < devices_numbers; i++){
		CUDACHECK(cudaSetDevice(devices[i]));
		CUDACHECK(cudaStreamDestroy(stream[i]));
	}
}

void init_NCCL(ncclComm_t *comms, int *devices, int devices_numbers){
  // *comms = (ncclComm_t *)malloc(sizeof(ncclComm_t) * devices_numbers);
  NCCLCHECK(ncclCommInitAll(comms, devices_numbers, devices));
}

void finish_NCCL(ncclComm_t *comms, int devices_numbers){
  for(int i = 0; i < devices_numbers; i++)
    NCCLCHECK(ncclCommDestroy(comms[i]));
}

void Synchronize_streams(cudaStream_t *stream, int *devices, int devices_numbers){
  for(int i = 0; i < devices_numbers; i++){
    CUDACHECK(cudaSetDevice(devices[i]));
    CUDACHECK(cudaStreamSynchronize(stream[i]));
  }
}

void NCCL_AllReduce(float** sendbuff, float** recvbuff, int size, 
        ncclComm_t *comms, cudaStream_t *stream, int devices_numbers){
  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < devices_numbers; ++i)
    NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
        comms[i], stream[i]));
  NCCLCHECK(ncclGroupEnd());
}
void display(const float *device_data, int dev_id, int size){
  printf("Display Device %d:\n", dev_id);
  CUDACHECK(cudaSetDevice(dev_id));
  float *host_buff;
  CUDACHECK(cudaHostAlloc(&host_buff, size * sizeof(float), cudaHostAllocDefault));
  CUDACHECK(cudaMemcpy(host_buff, device_data, size * sizeof(float), cudaMemcpyDeviceToHost));
  for(int i = 0; i < size; i++){
    printf("%f ",host_buff[i]);
  }
  printf("\n");
  CUDACHECK(cudaFreeHost(host_buff));
}

void create(int **a, int n){
  *a = (int *)malloc(sizeof(int) * n);
  for(int i = 0; i < n; i++){
    (*a)[i] = 1;
  }
}
void for_each(int *a, int n){
  for(int i = 0; i < n; i++){
    printf("%d ",a[i]);
  }
  printf("\n");
}
void show_int(int a){
  printf("the num is %d\n", a);
}

void show_array2D(float **a, int row, int col){
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){
      printf("%f ", a[i][j]);
    }
    printf("\n");
  }
}

int main(){

  // int nDev = 8;
  // int size = 16;
  // int devs[8] = {0, 1, 2, 3, 4, 5, 6, 7};
  // cudaStream_t *s;
  // ncclComm_t *comms;

  // //allocating and initializing device buffers
  // float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  // float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  // // cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  // for(int i = 0; i < nDev; ++i) {
  //   CUDACHECK(cudaSetDevice(devs[i]));
  //   CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
  //   CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
  // }

  // create_streams(&s, devs, nDev);
  
  // //init_data
  // for(int i = 0; i < nDev; i++){
	// 	CUDACHECK(cudaSetDevice(i));
	// 	array_set<<<(size + THREADS_PER_BLOCKS - 1)/THREADS_PER_BLOCKS, THREADS_PER_BLOCKS, 0, s[i]>>>(sendbuff[i], i, size);
  //   array_set<<<(size + THREADS_PER_BLOCKS - 1)/THREADS_PER_BLOCKS, THREADS_PER_BLOCKS, 0, s[i]>>>(recvbuff[i], 0, size);
  // }

  // init_NCCL(&comms, devs, nDev);
  // NCCL_AllReduce(sendbuff, recvbuff, size, comms, s, nDev);
  // Synchronize_streams(s, devs, nDev);
  
  // for(int i = 0; i < nDev; i++){
  //   display(recvbuff[i], devs[i], size);
  // }
  printf("comm_t size is %ld\n", sizeof(ncclComm_t*));
  printf("void* size is %ld\n", sizeof(void*));
  return 0;
}

