from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class PadOp(Op):
    def __call__(self, node_A, paddings, mode="CONSTANT", constant_values=0):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Pad(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "Pad"
        else:
            new_node.name = "pad" + str(new_node.id)
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
            from ..gpu_links import pad
            pad(input_vals[0], output_val, self.paddings,
                self.mode, self.constant_values, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_Pad']:
                from ..cpu_links import pad as cpu_pad
                from ..ndarray import numpyasdlarrayhandle
                in_arr = numpyasdlarrayhandle(input_vals[0])
                out_arr = numpyasdlarrayhandle(output_val)
                cpu_pad(in_arr, out_arr, self.paddings, self.mode, constant_values=0)
            else:
                output_val[:] = pad_np(input_vals[0], self.paddings, self.mode, constant_values=0)
        else:
            from ..gpu_links import pad
            pad(input_vals[0], output_val, self.paddings,
                self.mode, self.constant_values, stream_handle, node.profiler)


    def gradient(self, node, output_grad):
        return [pad_gradient_op(output_grad, self.paddings, self.mode)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        out_shape = list(input_shapes[0])
        pad_len = len(self.paddings)
        for i in range(4):
            if(i - (4 - pad_len) >= 0):
                out_shape[i] = out_shape[i] + self.paddings[i -
                                                            (4 - pad_len)][0] + self.paddings[i - (4 - pad_len)][1]
        return tuple(out_shape)


class Pad_GradientOp(Op):
    def __call__(self, node_A, paddings, mode="CONSTANT"):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.paddings = paddings
        self.mode = mode
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Pad_Gradient(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "Pad_Gradient"
        else:
            new_node.name = "Pad_Gradient" + str(new_node.id)
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
            from ..gpu_links import pad_gradient
            pad_gradient(input_vals[0], output_val, self.paddings, self.mode, stream_handle, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_Pad_Gradient']:
                from ..cpu_links import pad_gradient as cpu_pad_gradient
                from ..ndarray import numpyasdlarrayhandle
                in_arr = numpyasdlarrayhandle(input_vals[0])
                out_arr = numpyasdlarrayhandle(output_val)
                cpu_pad_gradient(in_arr, out_arr, self.paddings, self.mode)
            else:
                output_val[:] = pad_np_gradient(input_vals[0], self.paddings)
        else:
            from ..gpu_links import pad_gradient
            pad_gradient(input_vals[0], output_val, self.paddings, self.mode, stream_handle, node.profiler)


    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        out_shape = list(input_shapes[0])
        pad_len = len(self.paddings)
        for i in range(4):
            if(i - (4 - pad_len) >= 0):
                out_shape[i] = out_shape[i] - self.paddings[i -
                                                            (4 - pad_len)][0] - self.paddings[i - (4 - pad_len)][1]
        return tuple(out_shape)


def pad_op(node_A, paddings, mode="CONSTANT", constant_values=0):
    """Pad an input variable.

    Parameters:
    ----
    node_A : Node
        The Node to be padded.
    paddings : Node
        padding edge
    mode :
        CONSTANT/REFLECT/SYMMETRIC
    constant_values: scalar value
        padding values

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PadOp()(node_A, paddings, mode, constant_values)


def pad_gradient_op(node_A, paddings, mode="CONSTANT"):
    """Gradient node of pad operation.

    Parameters:
    ----
    node_A : Node
        The Node to be padded.
    paddings : Node
        padding edge
    mode :
        CONSTANT/REFLECT/SYMMETRIC

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Pad_GradientOp()(node_A, paddings, mode)

def pad_np(node_A, paddings, mode="constant", constant_values=0):
    import numpy as np
    return np.pad(node_A, paddings, mode=mode, constant_values=(constant_values,constant_values))

def pad_np_gradient(grad,paddings):
    slices = []
    for c in paddings:
        e = None if c[1] ==0 else -c[1]
        slices.append(slice(c[0],e))
    return grad[tuple(slices)]