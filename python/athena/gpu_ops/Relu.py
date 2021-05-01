from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Relu(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "Relu"
        else:
            new_node.name = "Relu" + str(new_node.id)
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
            from ..gpu_links import relu
            relu(input_vals[0], output_val, None, node.profiler)
    
    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlRelu']:
                from ..cpu_links import relu as cpu_relu
                from ..ndarray import numpyasdlarrayhandle
                in_arr = numpyasdlarrayhandle(input_vals[0])
                out_arr = numpyasdlarrayhandle(output_val)
                cpu_relu(in_arr, out_arr)
            else:
                output_val[:] = np.maximum(input_vals[0], 0)
        else:
            from ..gpu_links import relu
            relu(input_vals[0], output_val, stream_handle, None)


    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "ReluGradient(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "ReluGradient"
        else:
            new_node.name = "ReluGradient" + str(new_node.id)
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
            from ..gpu_links import relu_gradient
            relu_gradient(input_vals[0], input_vals[1], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000
    
    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 2
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlRelu_Gradient']:
                from ..cpu_links import relu_gradient as cpu_relu_gradient
                from ..ndarray import numpyasdlarrayhandle
                in_arr = numpyasdlarrayhandle(input_vals[0])
                in_grad = numpyasdlarrayhandle(input_vals[1])
                out_arr = numpyasdlarrayhandle(output_val)
                cpu_relu_gradient(in_arr, in_grad, out_arr)
            # heaviside function, 0.5 at x=0
            else:
                output_val[:] = (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]
        else:   
            from ..gpu_links import relu_gradient
            relu_gradient(input_vals[0], input_vals[1], output_val, stream_handle, None)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        return input_shapes[0]


def relu_op(node):
    """Rectified Linear Unit.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReluOp()(node)


def relu_gradient_op(node_A, node_B):
    """Computes the gradient of the ReLU function.  
    
    Parameters:
    ----
    node_A : Node
        Relu input.
    node_B : Node
        Previous gradient node.
    
    Returns:
    ----
    A new Node instance created by Op.

    """
    return ReluGradientOp()(node_A, node_B)
