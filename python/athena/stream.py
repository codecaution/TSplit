from __future__ import absolute_import

from ._base import _LIB, check_call, c_array
import ctypes
import numpy as np
import scipy.sparse
# from . import ndarray


class DLStream(ctypes.Structure):
    _fields_ = [("device_id", ctypes.c_int),
                ("handle", ctypes.c_void_p)]

DLStreamHandle = ctypes.POINTER(DLStream)

class Stream(ctypes.Structure):
    __slots__ = ["handle"]
    def __init__(self, handle):
        self.handle = handle
    def __del__(self):
        check_call(_LIB.DLStreamDestroy(self.handle))
    def sync(self):
        check_call(_LIB.DLStreamSync(self.handle))

def create_stream_handle(ctx):
    # assert ndarray.is_gpu_ctx(ctx)
    handle = DLStreamHandle()
    check_call(_LIB.DLStreamCreate(ctx.device_id, ctypes.byref(handle)))
    return Stream(handle)

class DLEvent(ctypes.Structure):
    _fields_ = [("device_id", ctypes.c_int),
                ("handle", ctypes.c_void_p)]

DLEventHandle = ctypes.POINTER(DLEvent)

class Event(ctypes.Structure):
    __slots__ = ["handle"]
    def __init__(self, handle):
        self.handle = handle
    def __del__(self):
        check_call(_LIB.DLEventDestroy(self.handle))
    def sync(self):
        check_call(_LIB.DLEventSync(self.handle))
    def query(self):
        return _LIB.DLEventQuery(self.handle)
    def record(self, stream):
        check_call(_LIB.DLEventRecord(stream.handle, self.handle))

def create_event_handle(ctx):
    # assert ndarray.is_gpu_ctx(ctx)
    handle = DLEventHandle()
    check_call(_LIB.DLEventCreate(ctx.device_id, ctypes.byref(handle)))
    return Event(handle)