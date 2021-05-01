from __future__ import absolute_import

from ._base import _LIB, check_call, c_array
import ctypes
import numpy as np
import scipy.sparse
from .stream import create_event_handle
MEMORY_MANAGE_RULE = 1
class DLContext(ctypes.Structure):
    """DL context strucure."""
    _fields_ = [("device_id", ctypes.c_int),
                ("device_type", ctypes.c_int)]

    MASK2STR = {
        1: 'cpu',
        2: 'gpu',
    }

    def __init__(self, device_id, device_type):
        super(DLContext, self).__init__()
        self.device_id = device_id
        self.device_type = device_type

    def __repr__(self):
        return "%s(%d)" % (
            DLContext.MASK2STR[self.device_type], self.device_id)


class DLArray(ctypes.Structure):
    """DLArray in C API"""
    _fields_ = [("data", ctypes.c_void_p),
                ("ctx", DLContext),
                ("ndim", ctypes.c_int),
                ("shape", ctypes.POINTER(ctypes.c_int64))]


DLArrayHandle = ctypes.POINTER(DLArray)


def cpu(dev_id=0):
    """Construct a CPU device
    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return DLContext(dev_id, 1)


def gpu(dev_id=0):
    """Construct a CPU device
    Parameters
    ----------
    dev_id : int, optional
        The integer device id
    """
    return DLContext(dev_id, 2)


def is_gpu_ctx(ctx):
    """Return if context is GPU context.
    Parameters
    ----------
    ctx : DLContext
        The query context
    """
    return ctx and ctx.device_type == 2


class NDArray(object):
    """Lightweight NDArray class of DL runtime.
    Strictly this is only an Array Container(a buffer object)
    No arthimetic operations are defined.
    """
    __slots__ = ["handle", "auto_deleted", "swap", "in_gpu",\
                "cpu2gpu_event", "gpu_event", "gpu2cpu_event",\
                "splitted", "sTensor", "split_number", "split_dimension"]

    # pylint: disable=no-member
    def __init__(self, handle, auto_deleted = True):
        """Initialize the function with handle
        Parameters
        ----------
        handle : DLArrayHandle
            the handle to the underlying C++ DLArray
        """
        self.handle = handle
        self.auto_deleted = auto_deleted
        
        ctx = self.handle.contents.ctx
        self.splitted = False
        self.sTensor = list()
        self.split_number = 1
        if is_gpu_ctx(ctx):
            self.cpu2gpu_event = create_event_handle(ctx)
            self.gpu_event = create_event_handle(ctx)
            self.gpu2cpu_event = create_event_handle(ctx)
        
        self.swap = False
        self.in_gpu = True
        self.split_dimension = None

    def __del__(self):
        # print '__del__ start'
        if self.auto_deleted == True:
            # from athena.gpu_ops.executor import MEMORY_MANAGE_RULE
            check_call(_LIB.DLArrayFree(self.handle, ctypes.c_int(MEMORY_MANAGE_RULE)))           
        return 

    def is_swapout_ok(self):
        
        return self.gpu2cpu_event.query()

    def is_swapin_ok(self):
        if self.cpu2gpu_event.query():
            self.in_gpu = True
            return 1
        else:
            return 0
    
    def split_tensor(self, total_number, dimension = "batch"):
        assert self.splitted == True
        if self.split_number == total_number and\
                self.split_dimension == dimension:
            return
        self.split_number = total_number
        input_tensor_handle = self.handle
        self.sTensor = []
        self.auto_deleted = False
        if dimension == "batch":
            dim = ctypes.c_int(0)
        elif dimension == "channel":
            dim = ctypes.c_int(1)
        else:
            raise NotImplementedError
        self.split_dimension = dimension
        for index in range(total_number):
            stensor_handle = DLArrayHandle()
            check_call(_LIB.DLArraySliceForMicroTensor(
                input_tensor_handle, ctypes.byref(stensor_handle),
                ctypes.c_int(total_number), ctypes.c_int(index), dim))
            stensor = NDArray(stensor_handle)
            stensor.in_gpu = True
            stensor.auto_deleted = False
            self.sTensor.append(stensor)
    
    def merge(self):
        assert self.splitted == False
        # self.sTensor = []
        # self.auto_deleted = True
        self.in_gpu = True


    # delete the data array by self
    def delete_itself(self):
        if self.in_gpu == True:
            self.auto_deleted = False
            self.in_gpu = False
            # from athena.gpu_ops.executor import MEMORY_MANAGE_RULE
            check_call(_LIB.DLArrayFreeSelf(self.handle, ctypes.c_int(MEMORY_MANAGE_RULE)))
            # print 'delete_itself end'
            return

    
    # malloc the data array by self
    def malloc_itself(self):
        # from .gpu_ops.executor import MEMORY_MANAGE_RULE
        check_call(_LIB.DLArrayAllocSelf(self.handle, ctypes.c_int(MEMORY_MANAGE_RULE)))
        self.in_gpu = True
        self.auto_deleted = True
        return 

    @property
    def shape(self):
        """Shape of this array"""
        return tuple(self.handle.contents.shape[i]
                     for i in range(self.handle.contents.ndim))

    @property
    def ctx(self):
        """context of this array"""
        return self.handle.contents.ctx

    def __setitem__(self, in_slice, value):
        """Set ndarray value"""
        if (not isinstance(in_slice, slice) or
                in_slice.start is not None
                or in_slice.stop is not None):
            raise ValueError('Array only support set from numpy array')
        if isinstance(value, NDArray):
            if value.handle is not self.handle:
                value.copyto(self)
        elif isinstance(value, (np.ndarray, np.generic)):
            self._sync_copyfrom(value)
        else:
            raise TypeError('type %s not supported' % str(type(value)))

    def _sync_copyfrom(self, source_array, data_type=np.float32):
        """Peform an synchronize copy from the array.
        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        if not isinstance(source_array, np.ndarray):
            try:
                source_array = np.array(source_array, dtype=data_type)
            except:
                raise TypeError('array must be an array_like data,' +
                                'type %s is not supported'
                                % str(type(source_array)))
        source_array = np.ascontiguousarray(source_array, dtype=data_type)
        if source_array.shape != self.shape:
            raise ValueError('array shape do not match the shape of NDArray')
        source_arr, shape = NDArray._numpyasarray(source_array)
        check_call(_LIB.DLArrayCopyFromTo(
            ctypes.byref(source_arr), self.handle, None))
        # de-allocate shape until now
        _ = shape

    def _async_copyfrom(self, source_array, stream, event=None):
        """Peform an asynchronize copy from the array.
        Parameters
        ----------
        source_array : array_like
            The data source we should like to copy from.
        """
        check_call(_LIB.DLArrayCopyFromTo(
            source_array.handle, self.handle, stream.handle))
        if not event is None:
            event.record(stream)

    def async_h2d(self, source_array, stream_handle, event_handle=None):
        assert self.handle.contents.ctx.device_type == 2
        assert source_array.handle.contents.ctx.device_type == 1
        assert stream_handle
        self._async_copyfrom(source_array, stream_handle, event_handle)

    def async_d2h(self, source_array, stream_handle, event_handle=None):
        assert self.handle.contents.ctx.device_type == 1
        assert source_array.handle.contents.ctx.device_type == 2
        assert stream_handle
        self._async_copyfrom(source_array, stream_handle, event_handle)
    
    def to_swap(self):
        self.swap = True

    @staticmethod
    def _numpyasarray(np_data):
        """Return a DLArray representation of a numpy array."""
        data = np_data
        assert data.flags['C_CONTIGUOUS']
        arr = DLArray()
        shape = c_array(ctypes.c_int64, data.shape)
        arr.data = data.ctypes.data_as(ctypes.c_void_p)
        arr.shape = shape
        arr.ndim = data.ndim
        # CPU device
        arr.ctx = cpu(0)
        return arr, shape

    def asnumpy(self):
        """Convert this array to numpy array
        Returns
        -------
        np_arr : numpy.ndarray
            The corresponding numpy array.
        """
        np_arr = np.empty(self.shape, dtype=np.float32)
        arr, shape = NDArray._numpyasarray(np_arr)
        check_call(_LIB.DLArrayCopyFromTo(
            self.handle, ctypes.byref(arr), None))
        _ = shape
        return np_arr

    def copyto(self, target, stream_handle = None):
        """Copy array to target
        Parameters
        ----------
        target : NDArray
            The target array to be copied, must have same shape as this array.
        """
        if isinstance(target, DLContext):
            target = empty(self.shape, target)
        if isinstance(target, NDArray):
            check_call(_LIB.DLArrayCopyFromTo(
                self.handle, target.handle, stream_handle.handle if stream_handle else None))
        else:
            raise ValueError("Unsupported target type %s" % str(type(target)))
        return target


def array(arr, ctx, stream_handle = None, event_handle = None):
    """Create an array from source arr.
    Parameters
    ----------
    arr : numpy.ndarray
        The array to be copied from
    ctx : DLContext, optional
        The device context to create the array
    Returns
    -------
    ret : NDArray
        The created array
    """
    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)
    ret = empty(arr.shape, ctx)
    ret._sync_copyfrom(arr)
    return ret


def empty(shape, ctx=cpu(0)):
    """Create an empty array given shape and device
    Parameters
    ----------
    shape : tuple of int
        The shape of the array
    ctx : DLContext
        The context of the array
    Returns
    -------
    arr : ndarray
        The array dlsys supported.
    """
    shape = c_array(ctypes.c_int64, shape)
    ndim = ctypes.c_int(len(shape))
    handle = DLArrayHandle()
    # from .gpu_ops.executor import MEMORY_MANAGE_RULE
    check_call(_LIB.DLArrayAlloc(
        shape, ndim, ctx, ctypes.byref(handle), ctypes.c_int(MEMORY_MANAGE_RULE)))
        
    return NDArray(handle)

def numpyasdlarrayhandle(data):
    if not data.flags['C_CONTIGUOUS']:
        data =  np.ascontiguousarray(data)
    arr = DLArray()
    shape = c_array(ctypes.c_int64, data.shape)
    arr.data = data.ctypes.data_as(ctypes.c_void_p)
    arr.shape = shape
    arr.ndim = data.ndim
    arr.ctx = cpu(0)
    return arr

def merge(output_vals, output_shape, ctx):
    for val in output_vals:
        del val
    return empty(output_shape, ctx = ctx)

def tensor_reuse(input_shape, output_shape, pieces, overlap_pieces, ctx = gpu(0)):
    input_handle = DLArrayHandle()
    input_shape = c_array(ctypes.c_int64, input_shape)
    input_ndim = ctypes.c_int(len(input_shape))

    output_handle = DLArrayHandle()
    output_shape = c_array(ctypes.c_int64, output_shape)
    output_ndim = ctypes.c_int(len(output_shape))
    # from .gpu_ops.executor import MEMORY_MANAGE_RULE
    check_call(_LIB.DLArrayReuseAlloc(
        ctx, ctypes.c_int(MEMORY_MANAGE_RULE), 
        input_shape, input_ndim, ctypes.byref(input_handle),
        output_shape, output_ndim, ctypes.byref(output_handle),
        ctypes.c_int(pieces), ctypes.c_int(overlap_pieces)
    ))
    input_arr = NDArray(input_handle)
    output_arr = NDArray(output_handle)
    input_arr.auto_deleted = False
    output_arr.auto_deleted = False
    return input_arr, output_arr

def slice_for_micro_tensor(input_tensor, index, total_number, dimension = "batch"):
    micro_tensor_handle = DLArrayHandle()
    input_tensor_handle = input_tensor.handle

    if dimension == "batch":
        dim = ctypes.c_int(0)
    elif dimension == "channel":
        dim = ctypes.c_int(1)
    else:
        raise NotImplementedError
    check_call(_LIB.DLArraySliceForMicroTensor(
        input_tensor_handle, ctypes.byref(micro_tensor_handle),
        ctypes.c_int(total_number), ctypes.c_int(index), dim
    ))
    micro_tensor = NDArray(micro_tensor_handle)
    micro_tensor.auto_deleted = False
    return micro_tensor

def split_shape(input_shape, total_number = 0, dimension = "batch"):
    assert total_number > 0
    output_shapes = list()
    output_shape1 = list(input_shape)
    output_shape2 = list(input_shape)
    if dimension == "batch":
        K = output_shape1[0] % total_number
        output_shape1[0] = output_shape1[0] // total_number
        output_shape2[0] = output_shape2[0] // total_number + 1
    elif dimension == "channel":
        K = output_shape1[1] % total_number
        output_shape1[1] = output_shape1[1] // total_number
        output_shape2[1] = output_shape2[1] // total_number + 1

    for idx in range(total_number):
        if idx + 1 <= K:
            output_shapes.append(output_shape2)
        else:
            output_shapes.append(output_shape1)
    return output_shapes


class ND_Sparse_Array(object):
    __slots__ = ["data", "row", "col", "nrow", "ncol"]

    def __init__(self, data, row, col, nrow, ncol):
        self.data = data
        self.row = row
        self.col = col
        self.nrow = nrow
        self.ncol = ncol
        
    @property
    def shape(self):
        """Shape of this array"""
        return tuple((self.nrow, self.ncol))


def sparse_array(values, indices, shape, ctx=cpu(0)):
    """Create an sparse array from source arrs.
    ----------
    values : numpy.ndarray
        The value array to be copied from
    indices : tuple(numpy.ndarray, numpy.ndarray)
        The index array to be copied from
    ctx : DLContext, optional
        The device context to create the array
    Returns
    -------
    ret : NDArray
        The created array
    """
    assert len(shape) == len(indices) == 2
    assert len(values) == len(indices[0]) == len(indices[1])
    assert isinstance(indices, tuple)
    mat = scipy.sparse.csr_matrix((values, indices), shape)
    values = mat.data
    rows = mat.indptr
    cols = mat.indices
    values_ret = empty(values.shape, ctx)
    values_ret._sync_copyfrom(values)
    row_ret = empty(rows.shape, ctx)
    row_ret._sync_copyfrom(rows, np.int32)
    col_ret = empty(cols.shape, ctx)
    col_ret._sync_copyfrom(cols, np.int32)
    return ND_Sparse_Array(values_ret, row_ret, col_ret, shape[0], shape[1])
