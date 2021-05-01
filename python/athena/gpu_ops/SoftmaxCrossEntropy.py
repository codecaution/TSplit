from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "SoftmaxCrossEntropyOp(%s,%s)" % (
                node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "SoftmaxCrossEntropyOp"
        else:
            new_node.name = "SoftmaxCrossEntropyOp" + str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def profile(self, node, input_vals, output_val, is_static = True):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
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
            from ..gpu_links import softmax_cross_entropy
            softmax_cross_entropy(y, y_, output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        if use_numpy:
            from .._base import DNNL_LIB
            from ..ndarray import numpyasdlarrayhandle
            if DNNL_LIB['DnnlSoftmaxCrossEntropy']:
                from ..cpu_links import softmax_crossentropy
                matA = numpyasdlarrayhandle(y)
                matB = numpyasdlarrayhandle(y_)
                matC = numpyasdlarrayhandle(output_val)
                softmax_crossentropy(matA, matB, matC)
            else:
                from .Softmax import softmax_func
                # print('y:',y[0][0])
                # print('y_:',y_[0])
                softmax = softmax_func(y)
                # print("softmax:",softmax[0])
                cross_entropy = np.mean(
                    -np.sum(y_ * np.log(softmax), axis=1), keepdims=True)
                output_val[:] = cross_entropy
        else:
            from ..gpu_links import softmax_cross_entropy
            softmax_cross_entropy(y, y_, output_val, stream_handle, None)


    def gradient(self, node, output_grad):
        from .Softmax import softmax_op
        from .ZerosLike import zeroslike_op

        grad_A = softmaxcrossentropy_gradient_op(node.inputs[0], node.inputs[1], output_grad)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        return (1,)

class SoftmaxCrossEntropyGradientOp(Op):
    def __call__(self, node_A, node_B, node_C):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B, node_C]

        if NAME_RULE == 0:
            new_node.name = "(%s,%s,%s)" % (node_A.name, node_B.name, node_C.name)
        elif NAME_RULE == 1:
            new_node.name = "SoftmaxCrossEntropyGradientOp"
        else:
            new_node.name = "SoftmaxCrossEntropyGradientOp"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s, %s)" % (node_A.name, node_B.name, node_C.name)
        
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()

        return new_node

    def profile(self, node, input_vals, output_val, is_static = True):
        assert len(input_vals) == 3
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape) + \
                                         get_array_memory(input_vals[1].shape) + \
                                         get_array_memory(input_vals[2].shape)
            # output memory
            node.profiler.output_memory = get_array_memory(output_val.shape)
            # no workspace
            node.profiler.workspace_memory = 0
            # execute time
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            import time
            start = time.time()
            from ..gpu_links import softmax_cross_entropy_gradient
            softmax_cross_entropy_gradient(input_vals[0], input_vals[1], input_vals[2], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000


    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 3
        if use_numpy:
            from .._base import DNNL_LIB
            from ..ndarray import numpyasdlarrayhandle
            if DNNL_LIB['DnnlSoftmaxCrossEntropy_Gradient']:
                print('No support for DnnlSoftmaxCrossEntropy_gradient')
            else:
                from .Softmax import softmax_func
                output_val[:] = (softmax_func(input_vals[0]) + -1 * input_vals[1]) * input_vals[2] / input_vals[0].shape[0]
        else:
            from ..gpu_links import softmax_cross_entropy_gradient
            softmax_cross_entropy_gradient(input_vals[0], input_vals[1], input_vals[2], output_val, stream_handle, None)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        # print(input_shapes[0])
        return input_shapes[0]


def softmaxcrossentropy_op(node_A, node_B):
    """Computes cross entropy loss for pre-softmax activations.
    
    Parameters:
    ----
    node_A : Node
        Predicted probability.
    node_B : Node
        Labels.

    Returns:
    ----
    A new Node instance created by Op.

    """

    return SoftmaxCrossEntropyOp()(node_A, node_B)

def softmaxcrossentropy_gradient_op(node_A, node_B, node_C):
    """ Computes gradients of softmax cross entropy loss .

    Parameters:

    node_A : Node
        Predicted probability.
    node_B : Node
        Labels.
    node_C : Node
         Output gradient 


    """
    return SoftmaxCrossEntropyGradientOp()(node_A, node_B, node_C)
