#include "../header/mpi_nccl_communication.h"

static const ncclDataType_t TYPE2TYPE_V1[] = {
    ncclChar,            // ncclInt8, ncclChar
    ncclUint8,           // ncclUint8
    ncclInt32,           // ncclInt32, ncclInt
    ncclUint32,          // ncclUint32
    ncclInt64,           // ncclInt64
    ncclUint64,          // ncclUint64
    ncclFloat16,         // ncclFloat16, ncclHalf
    ncclFloat32,         // ncclFloat32, ncclFloat
    ncclFloat64          // ncclFloat64, ncclDouble
};

ncclDataType_t _get_proper_datatype(int datatype) {
    return TYPE2TYPE_V1[datatype];
}
static const ncclRedOp_t TYPE2TYPE_V2[] = {
    ncclSum,
    ncclProd,
    ncclMax,
    ncclMin
};
ncclRedOp_t _get_proper_redop(int redop){
  return TYPE2TYPE_V2[redop];
}
void MPIInit(){
  // MPICHECK(MPI_Init(argc, &argv));
  MPICHECK(MPI_Init(NULL, NULL));
}

void MPIFinalize(){
  MPICHECK(MPI_Finalize());
}

void MPIGetComm(MPI_Comm *comm){
  *comm = MPI_COMM_WORLD;
}

void MPIBcast(void *buffer, int size, MPI_Datatype datatype, int root, MPI_Comm comm){
  MPICHECK(MPI_Bcast(buffer, size, datatype, root, comm));
}

void getMPICommRank(MPI_Comm *comm, int *myRank){
  MPICHECK(MPI_Comm_rank(*comm, myRank));
}

void getMPICommSize(MPI_Comm *comm, int *nRanks){
  MPICHECK(MPI_Comm_size(*comm, nRanks));
}

uint64_t getHostHash(const char *string){
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

void getHostName(char *hostname, int maxlen){
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++){
    if (hostname[i] == '.'){
      hostname[i] = '\0';
      return;
    }
  }
}

void getLocalRank(MPI_Comm *comm, int nRanks, int myRank, int *localRank){
  int _localRank = 0;
  uint64_t hostHashs[nRanks];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[myRank] = getHostHash(hostname);
  MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, *comm));
  for (int p=0; p<nRanks; p++) {
     if (p == myRank) break;
     if (hostHashs[p] == hostHashs[myRank]) (_localRank)++;
  }
  *localRank = _localRank;
}

void setDevice(int device_id){
  CUDACHECK(cudaSetDevice(device_id));
}

void getNcclUniqueId(ncclUniqueId* Id, MPI_Comm mpi_comm, int localRank){
  // CUDACHECK(cudaSetDevice(localRank));
  if(localRank == 0)NCCLCHECK(ncclGetUniqueId(Id));
  MPIBcast((void*)Id, sizeof(ncclUniqueId), MPI_BYTE, 0, mpi_comm);
  // MPI_Barrier(MPI_COMM_WORLD);
}

void initNcclCommRank(ncclComm_t *comm, int nranks, ncclUniqueId *commId, int rank, int localRank){
  // CUDACHECK(cudaSetDevice(localRank));
  NCCLCHECK(ncclCommInitRank(comm, nranks, *commId, rank));
}

void _ncclAllReduce(const void *sendbuff, void *recvbuff, int size, int datatype,
                    int op, ncclComm_t *comm, cudaStream_t stream){

  NCCLCHECK(ncclAllReduce((const void *)sendbuff, (void *)recvbuff, size, _get_proper_datatype(datatype), _get_proper_redop(op), *comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
}

void dlarrayAllReduce(DLArray *array, int datatype, int op,
                      ncclComm_t *comm, cudaStream_t *stream){
  // int dev_id = (array->ctx.device_id);
  // CUDACHECK(cudaSetDevice(dev_id));
  int size = 1;
  // cudaStream_t s;
  // CUDACHECK(cudaStreamCreate(&s));
  for(int i = 0; i < array->ndim; i++){
    size = size * array->shape[i];
  }
  float *data_buffer = (float*)(array->data);
  _ncclAllReduce(data_buffer, data_buffer, size, datatype, op, comm, *stream);
}
void commDestroyNccl(ncclComm_t *comm){
    NCCLCHECK(ncclCommDestroy(*comm));
}

void display(const float *device_data, int dev_id, int size){
  printf("Display Device %d:\n", dev_id);
  CUDACHECK(cudaSetDevice(dev_id));
  float *host_buff;
  CUDACHECK(cudaHostAlloc(&host_buff, size * sizeof(float), cudaHostAllocDefault));
  CUDACHECK(cudaMemcpy(host_buff, device_data, size * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < size; i++){
    printf("%f ", host_buff[i]);
  }
  printf("\n");
  CUDACHECK(cudaFreeHost(host_buff));
}

void print_array(float *array, int size){
  float *output;
  output = (float*)malloc(sizeof(float) * size);
  cudaMemcpy(output, array, size * sizeof(float), cudaMemcpyHostToHost);
  for(int i = 0; i < size; i++){
    printf("%f ", output[i]);
  }
  printf("\n");
}
/*
int main(int argc, char *argv[])
{
  MPIInit();
  MPI_Comm mpi_comm = MPI_COMM_WORLD; //int64
  int myrank, nranks, localRank;
  getMPICommRank(&mpi_comm, &myrank);
  getMPICommSize(&mpi_comm, &nranks);
  getLocalRank(&mpi_comm, nranks, myrank, &localRank);
  printf("localRank is %d\n",localRank);

  ncclUniqueId id;
  ncclComm_t comm;
  cudaStream_t s;
  getNcclUniqueId(&id, mpi_comm, localRank);
  CUDACHECK(cudaSetDevice(localRank));
  int size = 16;
  float *sendbuff, *recvbuff;
  float *input_data = (float *)malloc(size * sizeof(float));
  for(int i = 0; i < size; i++)input_data[i] = myrank;
  CUDACHECK(cudaMalloc(&sendbuff, size * sizeof(float)));
  CUDACHECK(cudaMalloc(&recvbuff, size * sizeof(float)));
  CUDACHECK(cudaMemcpy(sendbuff, input_data, size * sizeof(float), cudaMemcpyHostToDevice));
  CUDACHECK(cudaStreamCreate(&s));

  initNcclCommRank(&comm, nranks, &id, myrank);

  //communicating using NCCL
  _ncclAllReduce((const void*)sendbuff, (void*)sendbuff, size, ncclFloat, ncclSum, &comm, &s);

  //completing NCCL operation by synchronizing on the CUDA stream
  CUDACHECK(cudaStreamSynchronize(s));
  if(myrank == 1){
    printf("send buff is \n");
    print_array(sendbuff, size);
    printf("recv buff is \n");
    print_array(recvbuff, size);
  }
  //free device buffers
  CUDACHECK(cudaFree(sendbuff));
  CUDACHECK(cudaFree(recvbuff));

  //finalizing NCCL
  ncclCommDestroy(comm);
  //finalizing MPI
  MPICHECK(MPI_Finalize());

  // printf("[MPI Rank %d] Success \n", myrank);
  return 0;
}*/
