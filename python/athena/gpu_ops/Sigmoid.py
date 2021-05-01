from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class SigmoidOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
          new_node.name = "Sigmoid(%s)" % (node_A.name)
        elif NAME_RULE == 1:
          new_node.name = "Sigmoid"
        else:
          new_node.name = "Sigmoid" + str(new_node.id)
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
            from ..gpu_links import sigmoid
            sigmoid(input_vals[0], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlSigmoid']:
                from ..cpu_links import sigmoid as cpu_sigmoid
                from ..ndarray import numpyasdlarrayhandle
                in_arr = numpyasdlarrayhandle(input_vals[0])
                out_arr = numpyasdlarrayhandle(output_val)
                cpu_sigmoid(in_arr, out_arr)
            else:
                output_val[:] = 1.0/(1.0+1.0/np.exp(input_vals[0]))
        else:
            from ..gpu_links import sigmoid
            sigmoid(input_vals[0], output_val, stream_handle, None)

    def gradient(self, node, output_grad):
        #ds=s(1-s)
        grad_A=sigmoid_op(node.inputs[0])*(1+ -1*sigmoid_op(node.inputs[0]))
        return [grad_A*output_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


def sigmoid_op(node):
    """Calculate sigmoid of a matrix elementwisely.

    Parameters:
    ----
    node : Node
        Input variable.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return SigmoidOp()(node)
