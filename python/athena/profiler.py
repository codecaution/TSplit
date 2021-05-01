from __future__ import absolute_import

from ._base import _LIB, check_call, c_array
import ctypes
import numpy as np
import scipy.sparse

FLOPS_PER_SECOND = 1 / (2000 / (0.012 * 0.020 * 3.3))
PAGEIN_LATENCY = 0.109 * 10 ** (-3)
PAGEIN_THROUGHPUT = 4616.51 * 10 ** (3)
PAGEOUT_LATENCY = 0.113 * 10 ** (-3)
PAGEOUT_THROUGHPUT = 4440 * 10 ** (3)

BandWidth = 16.0 * 1024 / 1000
class Profiler(ctypes.Structure):

    _fields_ = [("time", ctypes.c_float),
                ("input_memory", (ctypes.c_float)),
                ("output_memory", (ctypes.c_float)),
                ("workspace_memory", (ctypes.c_float))]

ProfilerHandle = ctypes.POINTER(Profiler)

def CreateProfiler():
    p = ProfilerHandle()
    check_call(_LIB.ProfilerAlloc(ctypes.byref(p)))
    return p.contents

def PrintProfiler(name, p):
    assert isinstance(p, Profiler)
    print('node=%-40s time=%-12f ms swap_time=%-12f ms input_memory=%-4f MB output_memory=%-4f MB workspace_memory=%-4f MB'
          %(name, p.time, p.output_memory / BandWidth,p.input_memory, p.output_memory, p.workspace_memory))