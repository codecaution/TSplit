#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "../cuda_common/gpu_runtime.h"
#include "nccl.h"
#include "mpi.h"
#include "../common/dlarray.h"
#define THREADS_PER_BLOCKS 1024

#define MPICHECK(cmd) do{                           \
        int e=cmd;                                  \
        if(e!= MPI_SUCCESS) {                       \
        printf("Failed: MPI error %s:%d '%d'\n",    \
        __FILE__,__LINE__, e);                      \
        exit(1);                                    \
        }                                           \
}while(0)

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

extern "C"{
  void MPIInit();
  void MPIFinalize();
  void MPIGetComm(MPI_Comm *comm);
  // void MPIBcast(void *buffer, int size, MPI_Datatype datatype, int root, MPI_Comm comm);
  void getMPICommRank(MPI_Comm *comm, int *myRank);
  void getMPICommSize(MPI_Comm *comm, int *nRanks);
  void getLocalRank(MPI_Comm *comm, int nRanks, int myRank, int *localRank);
  // void getNcclUniqueId(ncclUniqueId* Id);
  void getNcclUniqueId(ncclUniqueId* Id, MPI_Comm mpi_comm, int localRank);
  void initNcclCommRank(ncclComm_t *comm, int nranks, ncclUniqueId *commId, int rank, int localRank);
  void dlarrayAllReduce(DLArray *array, int datatype, int op,
                      ncclComm_t *comm, cudaStream_t *stream);
  // void dlarrayAllReduce(DLArray *array, int datatype, int op,
  //                     ncclComm_t* comm, cudaStream_t* stream);
  // void dlarrayAllReduce(DLArray *array, ncclComm_t* comm, cudaStream_t* stream);
  void commDestroyNccl(ncclComm_t *comm);
  void setDevice(int device_id);
}

