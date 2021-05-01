from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class ReduceSumAxisZeroOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "ReduceSumAxisZero(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "ReduceSumAxisZero"
        else:
            new_node.name = "ReduceSumAxisZero" + str(new_node.id)
            new_node.desc = new_node.name + "(%s)" % node_A.name
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
            from ..gpu_links import reduce_sum_axis_zero
            reduce_sum_axis_zero(input_vals[0], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_ReduceSumAxisZero']:
                from ..cpu_links import reduce_sum_axis_zero as cpu_reduce_sum_axis_zero
                from ..ndarray import numpyasdlarrayhandle
                in_arr = numpyasdlarrayhandle(input_vals[0])
                out_arr = numpyasdlarrayhandle(output_val)
                cpu_reduce_sum_axis_zero(in_arr, out_arr)
            else:
                assert(isinstance(input_vals[0], np.ndarray))
                output_val[:] = np.sum(input_vals[0], axis=0)
        else:
            from ..gpu_links import reduce_sum_axis_zero
            # print("reduce_sum_axis_zero: ", input_vals[0].shape, output_val.shape)
            # print(node.inputs[0].name)
            reduce_sum_axis_zero(input_vals[0], output_val, stream_handle, None)


    def gradient(self, node, output_grad):
        from .Broadcast import broadcastto_op
        return [broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        if len(input_shape) == 1:
            return (1,)
        else:
            return input_shape[1:]


def reducesumaxiszero_op(node):
    """Creates a node that represents np.sum(node_A, axis=0).

    Parameters:
    ----
    node : Node
        The Node needed to be summed.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReduceSumAxisZeroOp()(node)
