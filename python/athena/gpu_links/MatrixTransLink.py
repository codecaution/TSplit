from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_transpose(in_mat, out_mat, perm, stream = None, profiler = None):
    assert isinstance(in_mat, _nd.NDArray)
    assert isinstance(out_mat, _nd.NDArray)
    pointer_func = ctypes.c_int * len(perm)
    pointer = pointer_func(*list(perm))
    _LIB.DLGpuTranspose(in_mat.handle, out_mat.handle, pointer, stream.handle if stream else None, ctypes.byref(profiler) if profiler else None)