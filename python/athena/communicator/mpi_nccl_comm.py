from ctypes import *
from athena import ndarray
from athena.stream import *
import numpy as np
from enum import Enum
import os

def _load_nccl_lib():
    """Load libary in build/lib."""
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "lib_mpi_nccl_runtime_api.so")
    lib = CDLL(path_to_so_file, RTLD_GLOBAL)
    return lib

lib_mpi_nccl = _load_nccl_lib()
# lib_mpi_nccl = CDLL("./lib_mpi_nccl_runtime_api.so", RTLD_GLOBAL)


class ncclDataType_t(Enum):
    ncclInt8       = 0
    ncclChar       = 0
    ncclUint8      = 1
    ncclInt32      = 2
    ncclInt        = 2
    ncclUint32     = 3
    ncclInt64      = 4
    ncclUint64     = 5
    ncclFloat16    = 6
    ncclHalf       = 6
    ncclFloat32    = 7
    ncclFloat      = 7
    ncclFloat64    = 8
    ncclDouble     = 8
    ncclNumTypes   = 9

class ncclRedOp_t(Enum):
    ncclSum        = 0
    ncclProd       = 1
    ncclMax        = 2
    ncclMin        = 3
    ncclNumOps     = 4

class ncclUniqueId(Structure):
    _fields_=[("internal", (c_int8 * 128))]

class MPI_NCCL_Communicator():
    
    def __init__(self):
        '''
            mpicomm: the MPI communicator, to use in MPI_Bcast, MPI_Reduce, MPI_Scatter, etc
            ncclcomm: the NCCL communicator, to use in ncclAllReduce ...
            nRanks: the total number of MPI threads
            myRanks: the rank in all MPI threads
            localRank: the rank among the MPI threads in this device
            ncclId: ncclGetUniqueId should be called once when creating a communicator 
                    and the Id should be distributed to all ranks in the communicator before calling ncclCommInitRank.
            stream: the stream for NCCL communication
        '''
        self.mpicomm = c_int64(0)
        self.ncclcomm = c_int64(0)
        self.nRanks = c_int32(0)
        self.myRank = c_int32(0)
        self.localRank = c_int32(-1)
        self.ncclId = ncclUniqueId()
        self.stream = None
        self.device_id = c_int(-1)
        self.MPI_Init()
    
    def MPI_Init(self):
        lib_mpi_nccl.MPIInit()

    def MPI_Finalize(self):
        lib_mpi_nccl.MPIFinalize()

    def MPIGetComm(self):
        lib_mpi_nccl.MPIGetComm(ctypes.byref(self.mpicomm))
    
    def streamInit(self):
        assert(self.device_id.value != -1)
        self.stream = create_stream_handle(ndarray.gpu(self.device_id.value))

    def MPI_Comm_rank(self):
        lib_mpi_nccl.getMPICommRank(ctypes.byref(self.mpicomm), ctypes.byref(self.myRank))
    
    def MPI_Comm_size(self):
        lib_mpi_nccl.getMPICommSize(ctypes.byref(self.mpicomm), ctypes.byref(self.nRanks))
    
    def getLocalRank(self):
        lib_mpi_nccl.getLocalRank(ctypes.byref(self.mpicomm), self.nRanks, self.myRank, ctypes.byref(self.localRank))
    
    def ncclGetUniqueId(self):
        lib_mpi_nccl.getNcclUniqueId(ctypes.byref(self.ncclId), self.mpicomm, self.localRank)

    # def MPIBcast(self, array, size, datatype, root):
    #     lib_mpi_nccl.MPIBcast(ctypes.byref(self.ncclId), c_int(size), c_int(datatype), c_int(root), self.mpicomm)

    def dlarrayNcclAllReduce(self, dlarray, datatype, reduceop):
        lib_mpi_nccl.dlarrayAllReduce(dlarray.handle, c_int(datatype.value), c_int(reduceop.value), ctypes.byref(self.ncclcomm), self.stream.handle.contents.handle)   
        # lib_mpi_nccl.dlarrayAllReduce(dlarray.handle, c_int64(0), c_int64(0), ctypes.byref(self.ncclcomm), ctypes.byref(self.stream))   
        
    def ncclCommInitRank(self):
        '''
            Use partial AllReduce to change here.
            self.nRanks is the number of threads to use ncclallreduce
            self.myRank is the rank among these threads. the value must in [0, self.nRank - 1]
        '''
        lib_mpi_nccl.initNcclCommRank(ctypes.byref(self.ncclcomm), self.nRanks, ctypes.byref(self.ncclId), self.myRank, self.localRank)
        # lib_mpi_nccl.initNcclCommRank(ctypes.byref(self.ncclcomm), c_int32(5), ctypes.byref(self.ncclId), self.myRank, self.localRank)
    
    def ncclCommDestroy(self):
        lib_mpi_nccl.commDestroyNccl(ctypes.byref(self.ncclcomm))
    
    def ncclSetDevice(self, device_id):
        self.device_id.value = device_id
        lib_mpi_nccl.setDevice(self.device_id.value)
    # def show_property(self):
    #     print("self.comms = ", self.comms)
    #     print("self.streams = ", self.streams)
    #     print "self.devs = "
    #     lib_nccl.for_each(self.devs, self.devs_number)
    #     print"self.devs_number = ", self.devs_number.value
    def ncclAllReduceInit(self):
        self.MPIGetComm()
        self.MPI_Comm_rank()
        self.MPI_Comm_size()
        self.getLocalRank()
        # if(self.localRank.value == xxx):
        #     self.device_id = xxx
        self.device_id.value = self.localRank.value
        self.ncclSetDevice(self.device_id.value)
        self.ncclGetUniqueId()  
        self.ncclCommInitRank()
        self.streamInit()     

    def ncclAllReduceFinish(self):
        self.MPI_Finalize()
def mpi_nccl_communicator():
    '''

    '''
    return MPI_NCCL_Communicator()

# NCCL_DEBUG=INFO mpirun --allow-run-as-root -np 4 python mpi_nccl_comm.py
if __name__ == "__main__":
    # t = mpi_nccl_communicator()
    # # t.MPI_Init()
    # t.MPIGetComm()
    # t.MPI_Comm_rank()
    # t.MPI_Comm_size()
    # t.getLocalRank()
    # t.ncclSetDevice(t.localRank.value)
    # t.ncclGetUniqueId()
    # # t.MPIBcast(t.ncclId, 128, ncclDataType_t.ncclFloat32.value, 0)
    # if(t.localRank.value in [0, 1, 2, 4, 5]):
    #     t.ncclCommInitRank()
    #     t.streamInit()
    #     size = 16
    #     arr = np.ones(size)*t.localRank.value
    #     print(arr)
    #     arr = ndarray.array(arr, ctx = ndarray.gpu(t.localRank.value))
    #     t.dlarrayNcclAllReduce(arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
    #     # t.MPI_Finalize()
    #     # print(ncclDataType_t.ncclFloat.value)
    #     print(arr.asnumpy())
    # # else:
    #     # t.MPI_Finalize()
    t = mpi_nccl_communicator()
    t.ncclAllReduceInit()

    arr = np.ones(16)*t.localRank.value
    print(arr)
    arr = ndarray.array(arr, ctx = ndarray.gpu(t.device_id.value))

    t.dlarrayNcclAllReduce(arr, ncclDataType_t.ncclFloat32, ncclRedOp_t.ncclSum)
    
    print(arr.asnumpy())

    t.ncclAllReduceFinish()

