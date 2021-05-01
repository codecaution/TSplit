from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
import ndarray
import numpy as np
from ..stream import create_event_handle

class SwapOutOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.profiler = None
        self.event = create_event_handle(ndarray.gpu(0))
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "SwapOut(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "SwapOut"
        else:
            new_node.name = "SwapOut" + str(new_node.id)
            new_node.desc = new_node.name + "(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_val, output_val, use_numpy=True, stream_handle=None):
        if node.profiler is not None:
            import time
            start = time.time()
        assert len(input_val) == 1
        if use_numpy:
            raise NotImplementedError
        else:
            output_val.async_d2h(input_val, stream_handle, self.event)
            del input_val

        if node.profiler is not None:
            node.profiler.time = (time.time() - start) * 1000


    def gradient(self, node, output_grad):
        return 

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1

        return input_shapes[0]

class SwapInOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.profiler = None
        self.event = create_event_handle(ndarray.gpu(0))
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "SwapIn(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "SwapIn"
        else:
            new_node.name = "SwapIn" + str(new_node.id)
            new_node.desc = new_node.name + "(%s)" % (node_A.name)
        return new_node

    def compute(self, node, input_val, output_val, use_numpy=True, stream_handle=None):
        if node.profiler is not None:
            import time
            start = time.time()
        assert len(input_val) == 1
        if use_numpy:
            raise NotImplementedError
        else:
            output_val.async_h2d(input_val, stream_handle, self.event)
            del input_val

        if node.profiler is not None:
            node.profiler.time = (time.time() - start) * 1000


    def gradient(self, node, output_grad):
        return 

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1

        return input_shapes[0]

def swap_out_op(node_A):
    """swap out the array from device(GPU) to host(CPU).

    Parameters:
    ----
    node_A : Node
        The node to be swapped out.
    Returns:
    ----
    A new Node instance created by Op.

    """
    return SwapOutOp()(node_A)

def swap_in_op(node_A):
    """swap in the array from host(CPU) to device(GPU).

    Parameters:
    ----
    node_A : Node
        The node to be swapped in.
    Returns:
    ----
    A new Node instance created by Op.

    """
    return SwapInOp()(node_A)