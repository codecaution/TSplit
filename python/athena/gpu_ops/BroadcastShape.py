from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class BroadcastShapeOp(Op):
    def __call__(self, node_A, shape, add_axes=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.shape = shape
        new_node.add_axes = add_axes
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "BroadcastShape(%s,%s)" % (
                node_A.name, str(shape))
        elif NAME_RULE == 1:
            new_node.name = "BroadcastShape"
        else:
            new_node.name = "BroadcastShape" + str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s,%s)" % (node_A.name, str(shape))
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
            from ..gpu_links import broadcast_shape
            broadcast_shape(input_vals[0], output_val, node.add_axes, None)
            node.profiler.time = (time.time() - start) * 1000
            
            node.profiler.input_memory = get_array_memory(input_vals[0].shape)
            node.profiler.output_memory = get_array_memory(output_val.shape)

    def compute(self, node, input_vals, output_val, use_numpy=True ,stream_handle=None):
        assert node.shape is not None
        assert len(input_vals) == 1
        if use_numpy:
            input_shape = list(node.shape)
            for i in range(len(input_shape)):
                if node.add_axes and i in node.add_axes:
                    input_shape[i] = 1
            output_val[:] = np.broadcast_to(input_vals[0].asnumpy().reshape(input_shape), node.shape)
        else:
            from ..gpu_links import broadcast_shape
            broadcast_shape(input_vals[0], output_val, node.add_axes, stream_handle)

    def gradient(self, node, output_grad):
        from .ReduceSum import reduce_sum_op
        self.grad_node = reduce_sum_op(output_grad, None, None)
        return [self.grad_node]

    def infer_shape(self, node, input_shapes):
        assert node.shape is not None
        assert len(input_shapes) == 1
        input_shape = list(input_shapes[0])
        output_shape = list(node.shape)
        assert len(input_shape) <= len(output_shape)
        diff = len(output_shape) - len(input_shape)
        if node.add_axes:
            assert diff == len(node.add_axes) or input_shape == [1]
            assert all([axis < len(output_shape) for axis in node.add_axes])
            in_ind = 0
            for i in range(len(output_shape)):
                if i not in node.add_axes:
                    assert input_shape[in_ind] == output_shape[i]
                    in_ind += 1
            if hasattr(self, 'grad_node'):
                self.grad_node.axes = tuple(node.add_axes)
                self.grad_node.axes.keepdims = [False] * len(node.add_axes)
        else:
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
        return tuple(output_shape)


def broadcast_shape_op(node_A, shape, add_axes=None):
    """Creates a node that represents np.broadcast_to(node_A, shape).
    Parameters:
    ----
    node_a : Node
        The Node to be broadcast.
    shape : tuple
        Target shape.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return BroadcastShapeOp()(node_A, shape, add_axes=add_axes)