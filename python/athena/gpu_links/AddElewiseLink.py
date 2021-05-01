from __future__ import absolute_import

import ctypes
from .._base import _LIB
from .. import ndarray as _nd


def matrix_elementwise_add(matA, matB, matC, stream = None, profiler = None):
    assert isinstance(matA, _nd.NDArray)
    assert isinstance(matB, _nd.NDArray)
    assert isinstance(matC, _nd.NDArray)
    _LIB.DLGpuMatrixElementwiseAdd(matA.handle, matB.handle, matC.handle, stream.handle if stream else None,  ctypes.byref(profiler) if profiler else None)
