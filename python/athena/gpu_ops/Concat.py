from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
import numpy as np
from .._base import get_array_memory
# Y = ad.concat_op(A, B, axis = 0)


class ConcatOp(Op):
    def __call__(self, node_A, node_B, axis=0):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        self.axis = axis
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Concat(%s, %s)" % (node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "Concat"
        else:
            new_node.name = "Concat" + str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (node_A.name, node_B.name)
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
            from ..gpu_links import concat
            concat(input_vals[0], input_vals[1], output_val, self.axis, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 2
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlConcat']:
                from ..cpu_links import concat as cpu_concat
                from ..ndarray import numpyasdlarrayhandle
                input_x = numpyasdlarrayhandle(input_vals[0])
                input_y = numpyasdlarrayhandle(input_vals[1])
                output = numpyasdlarrayhandle(output_val)
                cpu_concat(input_x, input_y, output, self.axis)
            else:
                output_val[:] = np.concatenate((input_vals[0], input_vals[1]),self.axis)
        else:
            from ..gpu_links import concat
            concat(input_vals[0], input_vals[1], output_val, self.axis, stream_handle, None)

    def gradient(self, node, output_grad):
        return [concat_gradient_op(output_grad, node.inputs[0], self.axis, idx=0),
                concat_gradient_op(output_grad, node.inputs[1], self.axis, idx=1)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        out_shape = list(input_shapes[0])
        out_shape[self.axis] = out_shape[self.axis] + \
            input_shapes[1][self.axis]

        return tuple(out_shape)


class Concat_gradientOP(Op):
    def __call__(self, grad_node, input_node, axis, idx):
        new_node = Op.__call__(self)
        new_node.inputs = [grad_node, input_node]
        self.axis = axis
        self.idx = idx
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Concat_gradient(%s, %s)" % (
                grad_node.name, input_node.name)
        elif NAME_RULE == 1:
            new_node.name = "Concat_gradient"
        else:
            new_node.name = "Concat_gradient" + str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (grad_node.name, input_node.name)
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
            from ..gpu_links import concat_gradient
            concat_gradient(input_vals[0], output_val, self.axis, self.idx, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 2
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_Concat_Gradient']:
                from ..cpu_links import concat_gradient as cpu_concat_gradient
                from ..ndarray import numpyasdlarrayhandle
                input = numpyasdlarrayhandle(input_vals[0])
                output = numpyasdlarrayhandle(output_val)
                cpu_concat_gradient(
                    input, output,self.axis, self.idx)
            else:
                output_val[:] = concat_backward(input_vals[0], output_val, self.axis)
        else:
            from ..gpu_links import concat_gradient
            concat_gradient(input_vals[0], output_val, self.axis, self.idx, stream_handle, None)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


def concat_op(node_A, node_B, axis=0):
    """Concatenates given variables along an axis.

    Parameters:
    ----
    node_A : Node
        The first node to be concated.
    node_B : Node
        The second node to be concated.
    axis :
        The axis along which two nodes are concated.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ConcatOp()(node_A, node_B, axis)


def concat_gradient_op(grad_node, input_node, axis, idx):
    """Gradient node of concat operation.

    Parameters:
    ----
    grad_node : Node
        Previous gradient node.
    input_node : Node
    axis :
        Axis along which to be concatenated.
    idx :
        The index of concatenation.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Concat_gradientOP()(grad_node, input_node, axis, idx)

def concat_backward(grad,input_nodes,axis=0):
    i1 = input_nodes[0].shape[axis]
    i2 = input_nodes[1].shape[axis]
    idx = min(i1,i2)
    if axis == 0:
        gradient_x1 = grad[:idx]
        gradient_x2 = grad[idx:]
    elif axis == 1:
        gradient_x1 = grad[:,:idx]
        gradient_x2 = grad[:,idx:]
    elif axis == 2:
        gradient_x1 = grad[:,:,:idx]
        gradient_x2 = grad[:,:,idx:]
    else:
        gradient_x1 = grad[:,:,:,:idx]
        gradient_x2 = grad[:,:,:,idx:]
    return [gradient_x1,gradient_x2]