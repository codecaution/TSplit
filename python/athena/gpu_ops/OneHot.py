from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class OneHotOp(Op):
    def __call__(self, node_A, num_classes):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.num_classes = num_classes
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "OneHot(%s, %d)" % (node_A.name, num_classes)
        elif NAME_RULE == 1:
            new_node.name = "OneHot"
        else:
            new_node.name = "OneHot" + str(new_node.id)
            new_node.desc = new_node.name + "(%s, %d)" % (node_A.name, num_classes)
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
            from ..gpu_links import one_hot
            one_hot(input_vals[0], output_val, None)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 1
        if use_numpy:
            inputs = input_vals[0].asnumpy().astype(np.int)
            res = np.eye(node.num_classes)[inputs.reshape(-1)]
            output_val[:] = res.reshape(list(inputs.shape) + [node.num_classes]).astype(np.float32)
        else:
            from ..gpu_links import one_hot
            # pass
            one_hot(input_vals[0], output_val, stream_handle)

    def gradient(self, node, output_grad):
        return [None]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return tuple(list(input_shapes[0]) + [node.num_classes])


def one_hot_op(node, num_classes):
    """Creates a node that represents one hot.
    Parameters:
    ----
    node : Node
        The input Node.
    num_classes: int
        Number of classes.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return OneHotOp()(node, num_classes)