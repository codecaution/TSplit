from __future__ import absolute_import

from ._base import _LIB
import ctypes

def get_gpu_memory():
    usage = {}
    usage['used_size'] = ctypes.c_uint64(0)
    usage['unused_unavailable_size'] = ctypes.c_uint64(0)
    usage['unused_available_size'] = ctypes.c_uint64(0)
    # print(usage)
    _LIB.DLMemoryManager_GPU_Usage(ctypes.byref(usage['used_size']),ctypes.byref(usage['unused_unavailable_size']), ctypes.byref(usage['unused_available_size']))
    return usage

def print_gpu_memory():
    usage = get_gpu_memory()
    print("Used: {} MB Unused available: {} MB".format(usage['used_size'].value / (2**20), usage['unused_available_size'].value / (2**20)))

def print_gpu_used_memory(phase = None):
    usage = get_gpu_memory()
    print("Phase: {} Used: {} MB".format(phase, usage['used_size'].value / (2**20)))

def judge_malloc(shape):
    num = 1
    for i in shape:
        num *= i
    memory_size = ctypes.c_uint64(num * 4)
    t = ctypes.c_int(0)
    _LIB.DLMemoryManager_Try_to_Malloc(memory_size, ctypes.byref(t))
    return t.value

def print_usage():
    t = ctypes.c_int(0)
    _LIB.DLMemoryManager_PrintState(ctypes.byref(t), ctypes.c_int(1))
    return t.value

def judge_usage():
    t = ctypes.c_int(0)
    _LIB.DLMemoryManager_PrintState(ctypes.byref(t), ctypes.c_int(0))
    return t.value