from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class WhereOp(Op):
    def __call__(self, cond, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [cond, node_A, node_B]
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()        
        if NAME_RULE == 0:
            new_node.name = "Where(%s, %s, %s)" % (cond.name, node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "Where"
        else:
            new_node.name = "Where" + str(new_node.id)
            new_node.desc = new_node.name + "(%s, %s, %s)" % (cond.name, node_A.name, node_B.name)
        return new_node
    def profile(self, node, input_vals, output_val, is_static = True):   
        assert len(input_vals) == 3
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape) +\
                                            get_array_memory(input_vals[1].shape) +\
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
            from ..gpu_links import where
            where(input_vals[0], input_vals[1], input_vals[2], output_val, None)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 3
        if use_numpy:
            output_val[:] = np.where(input_vals[0].asnumpy(), input_vals[1].asnumpy(), input_vals[2].asnumpy())
        else:
            from ..gpu_links import where
            # pass
            where(input_vals[0], input_vals[1], input_vals[2], output_val, stream_handle)
    
    def gradient(self, node, output_grad):
        from .ZerosLike import zeroslike_op
        zeros = zeroslike_op(node.inputs[0])
        grad_A = where_op(node.inputs[0], output_grad, zeros)
        grad_B = where_op(node.inputs[0], zeros, output_grad)
        return [None, grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        # print(input_shapes[0], input_shapes[1], input_shapes[2])
        assert tuple(input_shapes[0]) == tuple(input_shapes[1])
        assert tuple(input_shapes[1]) == tuple(input_shapes[2])
        return input_shapes[0]


def where_op(cond, node_A, node_B):
    """Creates a node that represents np.where.
    Parameters:
    ----
    cond : Node of a condition array
    node_A : Node, output if cond
    node_B : Node, output if not cond
    Returns:
    ----
    A new Node instance created by Op.
    """
    return WhereOp()(cond, node_A, node_B)