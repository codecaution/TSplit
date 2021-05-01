from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class ReduceSumOp(Op):
    def __call__(self, node_A, axes, keepdims=False):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if axes is not None:
            if isinstance(axes, int):
                axes = [axes]
            new_node.axes = list(axes)
            assert all(map(lambda x: isinstance(x, int), new_node.axes))        
        if keepdims is not None:
            if keepdims is True or keepdims is False:
                new_node.keepdims = [keepdims] * len(new_node.axes)
            else:
                keepdims = list(keepdims)
                assert len(keepdims) == len(new_node.axes)
                assert all(map(lambda x: isinstance(x, bool), keepdims))
                new_node.keepdims = keepdims
        if NAME_RULE == 0:
            new_node.name = "ReduceSum(%s, %s, %s)" % (node_A.name, str(new_node.axes), str(new_node.keepdims))
        elif NAME_RULE == 1:
            new_node.name = "ReduceSum"
        else:
            new_node.name = "ReduceSum" + str(new_node.id)
            new_node.desc = new_node.name + "(%s, %s, %s)" % (node_A.name, str(new_node.axes), str(new_node.keepdims))
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
            from ..gpu_links import reduce_sum
            reduce_sum(input_vals[0], output_val, node.axes, None)
            node.profiler.time = (time.time() - start) * 1000
 
    def compute(self, node, input_vals, output_val, use_numpy=True ,stream_handle=None):
        assert node.axes is not None and node.keepdims is not None
        assert len(input_vals) == 1
        if use_numpy:
            if all(node.keepdims) or not any(node.keepdims):
                output_val[:] = np.sum(input_vals[0].asnumpy(), axis=tuple(node.axes), keepdims=node.keepdims[0])
            else:
                temp = input_vals[0].asnumpy()
                for i in range(len(node.keepdims))[::-1]:
                    temp = np.sum(temp, node.axes[i], keepdims=node.keepdims[i])
                output_val[:] = temp
        else:
            from ..gpu_links import reduce_sum
            # pass
            reduce_sum(input_vals[0], output_val, node.axes, stream_handle)

    def gradient(self, node, output_grad):
        from .BroadcastShape import broadcast_shape_op
        self.grad_node = broadcast_shape_op(output_grad, None)
        return [self.grad_node]

    def infer_shape(self, node, input_shapes):
        assert node.axes is not None and node.keepdims is not None
        assert len(input_shapes) == 1
        input_shape = list(input_shapes[0])
        if hasattr(self, 'grad_node'):
            self.grad_node.shape = tuple(input_shape)
            add_axes = []
            for i in range(len(node.axes)):
                if not node.keepdims[i]:
                    add_axes.append(node.axes[i])
            self.grad_node.add_axes = add_axes
        for i in range(len(node.axes)):
            if node.axes[i] < 0:
                node.axes[i] += len(input_shape)
            assert 0 <= node.axes[i] < len(input_shape)
            input_shape[node.axes[i]] = 1 if node.keepdims[i] else 0
        input_shape = [x for x in input_shape if x > 0]
        if input_shape == []:
            return (1,)
        else:
            return tuple(input_shape)


def reduce_sum_op(node, axes, keepdims=False):
    """Creates a node that represents np.sum(node_A, axis, keepdims).
    Parameters:
    ----
    node : Node
        The Node needed to be summed.
    axes : int or list
        The axis/axes needed to be summed.
    keepdims: bool or list
        Whether to keep the dimension(s).
    Returns:
    ----
    A new Node instance created by Op.
    """
    return ReduceSumOp()(node, axes, keepdims)