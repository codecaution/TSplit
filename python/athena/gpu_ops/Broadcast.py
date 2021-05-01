from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class BroadcastToOp(Op):
    def __call__(self, node_A, shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.profiler = None
        new_node.new_shape = shape
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "BroadcastTo(%s,%s.shape)" % (
                node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "BroadcastTo"
        else:
            new_node.name = "BroadcastTo" + str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s,%s.shape)" % (node_A.name, node_B.name)
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
            from ..gpu_links import broadcast_to
            broadcast_to(input_vals[0], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000
    
    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert(len(input_vals) == 1)
        # print node.inputs[0].name, node.inputs[1].name
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_BroadcastTo']:
                from ..cpu_links import broadcast_to as cpu_broadcast_to
                from ..ndarray import numpyasdlarrayhandle
                input = numpyasdlarrayhandle(input_vals[0])
                output = numpyasdlarrayhandle(output_val)
                cpu_broadcast_to(
                    input, output)
            else:
                output_val[:] = np.broadcast_to(input_vals[0], node.new_shape)
        else:
            from ..gpu_links import broadcast_to
            broadcast_to(input_vals[0], output_val, stream_handle, None)


    def gradient(self, node, output_grad):
        from .ReduceSumAxisZero import reducesumaxiszero_op
        from .ZerosLike import zeroslike_op

        grad_A = reducesumaxiszero_op(output_grad)
        return [grad_A]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return node.new_shape


def broadcastto_op(node_A, shape):
    """Creates a node that represents np.broadcast_to(node_A, node_B.shape).

    Parameters:
    ----
    node_a : Node
        The Node to be broadcast.
    shape

    Returns:
    ----
    A new Node instance created by Op.

    """
    return BroadcastToOp()(node_A, shape)
