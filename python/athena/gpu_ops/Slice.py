from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class SliceOp(Op):
    def __call__(self, node_A, begin_pos, output_shape):
        new_node = Op.__call__(self)
        new_node.begin_pos = tuple(begin_pos)
        new_node.output_shape = list(output_shape)
        assert len(new_node.begin_pos) == len(new_node.output_shape)
        for i in range(len(new_node.begin_pos)):
            assert new_node.begin_pos[i] >= 0
        new_node.inputs = [node_A]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Slice(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "Slice"
        else:
            new_node.name = "Slice" + str(new_node.id)
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
            from ..gpu_links import matrix_slice
            matrix_slice(input_vals[0], output_val, node.begin_pos, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 1
        ori_shape = list(input_vals[0].shape)
        assert len(ori_shape) == len(node.begin_pos)
        for i in range(len(ori_shape)):
            assert node.begin_pos[i] + node.output_shape[i] <= ori_shape[i]
        if use_numpy:
            assert(isinstance(input_vals[0], np.ndarray))
            index = tuple([slice(i, i+j) for i, j in zip(node.begin_pos, node.output_shape)])
            output_val[:] = input_vals[0][index]
        else:
            from ..gpu_links import matrix_slice
            # pass
            matrix_slice(input_vals[0], output_val, node.begin_pos, stream_handle, None)

    
    def gradient(self, node, output_grad):
        self.grad_node = slice_gradient_op(output_grad, node.begin_pos)
        return [self.grad_node]
    
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])
        assert len(ori_shape) == len(node.begin_pos)
        for i in range(len(ori_shape)):
            if node.output_shape[i] == -1:
                node.output_shape[i] = ori_shape[i] - node.begin_pos[i]
            assert node.output_shape[i] > 0
            assert node.begin_pos[i] + node.output_shape[i] <= ori_shape[i]
        self.ori_shape = tuple(ori_shape)
        if hasattr(self, 'grad_node'):
            self.grad_node.output_shape = self.ori_shape
            assert len(self.ori_shape) == len(self.grad_node.begin_pos)
        node.output_shape = tuple(node.output_shape)
        return node.output_shape


class SliceGradientOp(Op):
    def __call__(self, node_A, begin_pos, output_shape):
        new_node = Op.__call__(self)
        new_node.begin_pos = tuple(begin_pos)
        new_node.output_shape = None
        if output_shape != None:
            new_node.output_shape = tuple(output_shape)
            assert len(new_node.begin_pos) == len(new_node.output_shape)
        for i in range(len(new_node.begin_pos)):
            assert new_node.begin_pos[i] >= 0
        new_node.inputs = [node_A]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "SliceGradient(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "SliceGradient"
        else:
            new_node.name = "SliceGradient" + str(new_node.id)
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
            from ..gpu_links import matrix_slice_gradient
            matrix_slice_gradient(input_vals[0], output_val, node.begin_pos, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000
    
    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert node.output_shape != None
        assert len(input_vals) == 1
        ori_shape = list(input_vals[0].shape)
        assert len(ori_shape) == len(node.begin_pos)
        for i in range(len(ori_shape)):
            assert node.begin_pos[i] + ori_shape[i] <= node.output_shape[i]
        if use_numpy:
            assert(isinstance(input_vals[0], np.ndarray))
            output_val[:] = np.zeros(node.output_shape, dtype=np.float32)
            index = tuple([slice(i, i+j) for i, j in zip(node.begin_pos, ori_shape)])
            output_val[index] = input_vals[0]
        else:
            from ..gpu_links import matrix_slice_gradient
            matrix_slice_gradient(input_vals[0], output_val, node.begin_pos, stream_handle, None)

    
    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert node.output_shape != None
        assert len(input_shapes) == 1
        ori_shape = list(input_shapes[0])
        assert len(ori_shape) == len(node.begin_pos)
        for i in range(len(ori_shape)):
            assert node.begin_pos[i] + ori_shape[i] <= node.output_shape[i] 
        return node.output_shape


def slice_op(node, begin, size):
    """Creates a node that represents tf.slice(node, begin, size).

    Parameters:
    ----
    node : Node
        The Node needed to be summed.
    begin: tuple
        The beginning position of slice operation.
    size: tuple
        The shape(size) of output tensor.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SliceOp()(node, begin, size)


def slice_gradient_op(node, begin, size=None):
    """Creates a node that represents the gradient of tf.slice.

    Parameters:
    ----
    node : Node
        The Node needed to be summed.
    begin: tuple
        The beginning position of slice operation.
    size: tuple
        The shape(size) of output tensor.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SliceGradientOp()(node, begin, size)
