from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class TransposeOp(Op):
    def __call__(self, node_A, perm=None):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.perm = perm
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()

        if NAME_RULE == 0:
          new_node.name = "Transpose(%s)" % (node_A.name)
        elif NAME_RULE == 1:
          new_node.name = "Transpose"
        else:
          new_node.name = "Transpose" + str(new_node.id)
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
            from ..gpu_links import matrix_transpose
            matrix_transpose(input_vals[0], output_val, node.perm, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        from ..gpu_links import matrix_transpose
        assert len(input_vals) == 1
        if node.perm is None:
            node.perm = list(range(len(input_vals[0].shape))[::-1])
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_Transpose']:
                from ..cpu_links import transpose as cpu_transpose
                cpu_transpose(input_vals[0], output_val, node.perm)
            else:
                output_val[:] = np.transpose(input_vals[0].asnumpy(), node.perm)
        else:
            # pass
            matrix_transpose(input_vals[0], output_val, node.perm, stream_handle, None)

    def gradient(self, node, output_grad):
        grad_perm = [0 for _ in node.perm]
        for i in range(len(node.perm)):
            grad_perm[node.perm[i]] = i
        return [transpose_op(output_grad, grad_perm)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        # only support matrix transpose
        # assert len(input_shapes[0]) == 2
        ori_shape = list(input_shapes[0])
        if node.perm is None:
            res_shape = ori_shape[::-1]
        else:
            assert len(node.perm) == len(ori_shape) and set(node.perm) == set(range(len(node.perm)))
            res_shape = [ori_shape[node.perm[i]] for i in range(len(ori_shape))]
        return res_shape

def transpose_op(node_A, perm=None):
    """Make a new instance of transpose and call the instance.
    Parameters:
    ----
    node_A : Node
        Node to be transposed.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return TransposeOp()(node_A, perm)