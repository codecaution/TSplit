from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class ZerosLikeOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Zeroslike(%s)" % node_A.name
        elif NAME_RULE == 1:
            new_node.name = "Zeroslike"
        else:
            new_node.name = "Zeroslike" + str(new_node.id)
            new_node.desc = new_node.name + "(%s)" % new_node.name
        return new_node
    
    def profile(self, node, input_vals, output_val, is_static = True):

        assert len(input_vals) == 1
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape)
            # output memory
            node.profiler.output_memory = get_array_memory(output_val.shape)
            # no workspace
            node.profiler.workspace_memory = 0
            # execute time
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            import time
            start = time.time()
            from ..gpu_links import array_set
            array_set(output_val, 0, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            from ..ndarray import numpyasdlarrayhandle
            if DNNL_LIB['cpu_ArraySet']:
                from ..cpu_links import array_set as cpu_array_set
                output_val[:]=np.empty(output_val.shape,dtype=np.float32)
                matA = numpyasdlarrayhandle(output_val)
                cpu_array_set(matA, 0)
            else:
                output_val[:] = np.zeros(input_vals[0].shape)
        else:
            from ..gpu_links import array_set
            array_set(output_val, 0, stream_handle, None)


    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


def zeroslike_op(node):
    """Creates a node that represents np.zeros(node_A.shape).

    Parameters:
    ----
    node : Node
        The Node to pad with 0.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ZerosLikeOp()(node)
