from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def broadcast_shape(in_arr, out_arr, add_axes=None, stream = None):
    assert isinstance(in_arr, _nd.NDArray)
    assert isinstance(out_arr, _nd.NDArray)
    if add_axes is not None:
        pointer_func = ctypes.c_int * len(add_axes)
        pointer = pointer_func(*list(add_axes))
    _LIB.DLGpuBroadcastShape(in_arr.handle, out_arr.handle, pointer if add_axes else None, stream.handle if stream else None)