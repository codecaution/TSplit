from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
import ctypes

class DropoutOp(Op):
    def __call__(self, node_in, keep_prob):
        new_node = Op.__call__(self)
        new_node.inputs = [node_in]
        new_node.reserve_size = ctypes.c_int(0)
        new_node.reserve_space = ctypes.c_void_p(0)
        new_node.mask = None
        self.keep_prob = keep_prob
        self.flag = 0
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s,%s)" % (node_in.name, keep_prob.name)
        elif NAME_RULE == 1:
            new_node.name = "DropoutOp"
        else:
            new_node.name = "DropoutOp"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (node_in.name, keep_prob.name)
        return new_node

    def profile(self, node, input_vals, output_val, is_static = True):
        assert len(input_vals) == 1
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape)
            # output memory
            node.profiler.output_memory = get_array_memory(output_val.shape)
            # no workspace
            # TODO
            node.profiler.workspace_memory = 0
            # execute time
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            import time
            start = time.time()
            from ..gpu_links import CuDNN_Dropout
            if self.flag == 0:
                CuDNN_Dropout(input_vals[0], self.keep_prob, output_val, node.reserve_size, node.reserve_space, 1, None, node.profiler)
                self.flag = 1
            else:
                CuDNN_Dropout(input_vals[0], self.keep_prob, output_val, node.reserve_size, node.reserve_space, 0, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_Dropout']:
                from ..cpu_links import dropout as cpu_dropout
                from ..ndarray import numpyasdlarrayhandle
                input_x = numpyasdlarrayhandle(input_vals[0])
                output = numpyasdlarrayhandle(output_val)
                cpu_dropout(input_x, self.keep_drop, output)
            else:
                import numpy as np
                np.random.seed(0.1)
                if node.mask is None:
                    node.mask = np.random.binomial(1, self.keep_drop, size=input_vals[0].shape)
                output_val[:] = dropout_np(input_vals[0], self.keep_drop, output_val, node.mask)
        else:
            from ..gpu_links import CuDNN_Dropout
            if self.flag == 0:
                CuDNN_Dropout(input_vals[0], self.keep_prob, output_val, node.reserve_size, node.reserve_space, 1,stream_handle, None)
                self.flag = 1
            else:
                CuDNN_Dropout(input_vals[0], self.keep_prob, output_val, node.reserve_size, node.reserve_space, 0,stream_handle, None)

    def gradient(self, node, output_grad):
        return [dropout_gradient_op(output_grad, self.keep_prob, node)]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class Dropout_GradientOp(Op):
    def __call__(self, node_in, keep_prob, forward_node):
        new_node = Op.__call__(self)
        new_node.inputs = [node_in]
        self.forward_node = forward_node
        self.keep_prob = keep_prob
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s,%s)" % (node_in.name, keep_prob.name)
        elif NAME_RULE == 1:
            new_node.name = "Dropout_GradientOp"
        else:
            new_node.name = "Dropout_GradientOp"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (node_in.name, keep_prob.name)
        return new_node

    def profile(self, node, input_vals, output_val, is_static = True):
        assert len(input_vals) == 1
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape)
            # output memory
            node.profiler.output_memory = get_array_memory(output_val.shape)
            # no workspace
            # TODO
            node.profiler.workspace_memory = 0
            # execute time
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            import time
            start = time.time()
            from ..gpu_links import CuDNN_Dropout_gradient
            CuDNN_Dropout_gradient(input_vals[0], self.keep_prob, output_val, self.forward_node.reserve_size,
                                self.forward_node.reserve_space, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000


    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_Dropout_Gradient']:
                from ..cpu_links import dropout_gradient as cpu_dropout_gradient
                from ..ndarray import numpyasdlarrayhandle
                input_x = numpyasdlarrayhandle(input_vals[0])
                output = numpyasdlarrayhandle(output_val)
                cpu_dropout_gradient(input_x, self.keep_drop, output)
            else:
                output_val[:] = dropout_np_gradient(input_vals[0], self.keep_drop, self.forward_node.mask)
        else:
            from ..gpu_links import CuDNN_Dropout_gradient
            CuDNN_Dropout_gradient(input_vals[0], self.keep_prob, output_val, self.forward_node.reserve_size,
                                   self.forward_node.reserve_space, stream_handle, None)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


def dropout_op(node_in, keep_prob):
    """Drops elements of input variable randomly.

    Parameters:
    ----
    node_in : Node
        Input variable.
    keep_prob : float
        Probability of the results to be kept.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DropoutOp()(node_in, keep_prob)


def dropout_gradient_op(node_in, keep_prob, forward_node):
    """Gradient node of dropout operation.

    Parameters:
    ----
    node_in : Node
        Input variable.
    keep_prob : float
        Probability of the results to be kept.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Dropout_GradientOp()(node_in, keep_prob, forward_node)

import numpy as np

def dropout_np(inputs, keep_drop, out_arr, mask):
    outputs = inputs
    outputs *= mask
    outputs = outputs * (1 / keep_drop)
    return outputs

def dropout_np_gradient(in_gradient_y, keep_drop,mask):
    out_grads = in_gradient_y
    out_grads *= mask * (1 / keep_drop)
    return out_grads