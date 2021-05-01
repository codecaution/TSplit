from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class BroadcastToTFOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
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
        new_node.inplace = False
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
            from ..gpu_links import broadcast_shape
            if node.inplace:
                input_vals[0].broadcast_to(input_vals[1].shape, output_val)
            else:
                broadcast_shape(input_vals[0], output_val, None, None)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy = True, stream_handle=None):
        assert len(input_vals) == 2
        if use_numpy:
            input_shape = list(input_vals[1].shape)
            output_val[:] = np.broadcast_to(input_vals[0].asnumpy(), input_shape)
        else:
            from ..gpu_links import broadcast_shape
            # print("broadcast to, ", input_vals[0].shape, output_val.shape)
            if node.inplace:
                input_vals[0].broadcast_to(input_vals[1].shape, output_val)
            else:
                # pass
                broadcast_shape(input_vals[0], output_val, None, stream_handle)

    def gradient(self, node, output_grad):
        from .ReduceSum import reduce_sum_op
        # from .ReduceSumAxisZero import reducesumaxiszero_op
        from .ZerosLike import zeroslike_op
        self.grad_node = reduce_sum_op(output_grad, None, None)
        # self.grad_node = reducesumaxiszero_op(output_grad)
        return [self.grad_node, None]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        input_shape = list(input_shapes[0])
        input_shape = list(input_shapes[0])
        output_shape = list(input_shapes[1])
        assert len(input_shape) <= len(output_shape)
        diff = len(output_shape) - len(input_shape)
        axes = list(range(diff))
        keepdims = [False] * diff
        input_shape = [1] * diff + input_shape
        for i in range(len(output_shape)):
            assert output_shape[i] > 0 and isinstance(output_shape[i], int)
            assert input_shape[i] == 1 or input_shape[i] == output_shape[i]
            if i >= diff and input_shape[i] == 1 and output_shape[i] > 1:
                axes.append(i)
                keepdims.append(True)
        if hasattr(self, 'grad_node'):
            self.grad_node.axes = axes
            self.grad_node.keepdims = keepdims
        return input_shapes[1]


def broadcasttoTF_op(node_A, node_B):
    """Creates a node that represents np.broadcast_to(node_A, node_B.shape).
    Parameters:
    ----
    node_a : Node
        The Node to be broadcast.
    node_b : Node
        Another Node with the target shape.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return BroadcastToTFOp()(node_A, node_B)