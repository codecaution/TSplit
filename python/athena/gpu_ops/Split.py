from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class SplitOp(Op):
    def __call__(self, node_A, sub_index, total_num):
        new_node = Op.__call__(self)
        new_node.sub_index = sub_index
        new_node.total_num = total_num

        new_node.inputs = [node_A]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Split(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "Split"
        else:
            new_node.name = "Split" + str(new_node.id)
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
            from ..gpu_links import array_split
            array_split(input_vals[0], output_val, node.sub_index, node.total_num, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 1

        if use_numpy:
            raise NotImplementedError
        else:
            from ..gpu_links import array_split
            array_split(input_vals[0], output_val, node.sub_index, node.total_num, stream_handle, None)

    
    def gradient(self, node, output_grad):
        self.grad_node = slice_gradient_op(output_grad, node.begin_pos)
        return [self.grad_node]
    
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        out_shape = list(input_shapes[0])
        out_shape[0] = out_shape[0] / node.total_num
        
        return tuple(out_shape)

def split_op(node, sub_index, total_num):
    '''Split the input node
    '''
    SplitOp()(node, sub_index, total_num)
