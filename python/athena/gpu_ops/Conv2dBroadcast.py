from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class Conv2d_BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Conv2d_BroadcastTo(%s,%s.shape)" % (
                node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "Conv2d_BroadcastTo"
        else:
            new_node.name = "Conv2d_BroadcastTo" + str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def profile(self, node, input_vals, output_val, is_static = True):

        assert len(input_vals) == 2
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape) + \
                                         get_array_memory(input_vals[1].shape)
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
        assert(len(input_vals) == 2)
        # print node.inputs[0].name, node.inputs[1].name
        if use_numpy:
            shapeW = input_vals[1].shape
            shapeW = list(shapeW)
            tmp = shapeW[1]
            shapeW[1] = shapeW[3]
            shapeW[3] = tmp
            output_val[:] = np.broadcast_to(
                input_vals[0], input_vals[1].shape).swapaxes(1, 3)
        else:
            from ..gpu_links import broadcast_to
            broadcast_to(input_vals[0], output_val, stream_handle, None)

    def gradient(self, node, output_grad):
        from .Conv2dReduceSum import conv2d_reducesum_op
        from .ZerosLike import zeroslike_op

        grad_A = conv2d_reducesum_op(output_grad)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        return input_shapes[1]


def conv2d_broadcastto_op(node_A, node_B):
    """Creates a node that represents np.broadcast_to(node_A, node_B.shape).

    Parameters:
    ----
    node_a : Node
        The Node to be bcast.
    node_b : Node
        Another Node with the target shape.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2d_BroadcastToOp()(node_A, node_B)
