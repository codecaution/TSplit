from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory


class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        elif NAME_RULE == 1:
            new_node.name = "AddbyConst"
        else:
            new_node.name = "AddbyConst"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (node_A.name, str(const_val))
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
            from ..gpu_links import matrix_elementwise_add_by_const
            matrix_elementwise_add_by_const(
                input_vals[0], node.const_attr, output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlMatrixElementwiseAddByConst']:
                from ..ndarray import numpyasdlarrayhandle
                from ..cpu_links import matrix_elementwise_add_by_const as cpu_matrix_elementwise_add_by_const
                matA = numpyasdlarrayhandle(input_vals[0])
                matC = numpyasdlarrayhandle(output_val)
                cpu_matrix_elementwise_add_by_const(matA, node.const_attr, matC)
            else:
                output_val[:] = input_vals[0] + node.const_attr
        else:
            from ..gpu_links import matrix_elementwise_add_by_const
            matrix_elementwise_add_by_const(
                input_vals[0], node.const_attr, output_val, stream_handle, None)

    def gradient(self, node, output_grad):
        return [output_grad]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        return input_shapes[0]


def addbyconst_op(node, const_val):
    """Make a new instance of AddByConstOp and call the instance.

    Parameters:
    ----
    node : Node
        The Node to be added.
    const_val : scalar value
        The constant value to be added.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AddByConstOp()(node, const_val)
