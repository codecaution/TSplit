from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory


class DivOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        if NAME_RULE == 0:
            new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "Div"
        else:
            new_node.name = "Div" + str(new_node.id)
            new_node.desc = new_node.name + "(%s, %s)" % (node_A.name, node_B.name)
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
            from ..gpu_links import matrix_elementwise_divide
            matrix_elementwise_divide(
                input_vals[0], input_vals[1], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 2
        assert input_vals[0].shape == input_vals[1].shape, \
            "can't do elementwise division between variables of different sizes."
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlMatrixElementwiseDivide']:
                from ..ndarray import numpyasdlarrayhandle
                from ..cpu_links import matrix_elementwise_divide as cpu_matrix_elementwise_divide
                matA = numpyasdlarrayhandle(input_vals[0])
                matB = numpyasdlarrayhandle(input_vals[1])
                matC = numpyasdlarrayhandle(output_val)
                cpu_matrix_elementwise_divide(matA, matB, matC)
            else:
                output_val[:] = input_vals[0] / input_vals[1]
        else:
            from ..gpu_links import matrix_elementwise_divide
            matrix_elementwise_divide(
                input_vals[0], input_vals[1], output_val, stream_handle, None)


    def gradient(self, node, output_grad):
        dividend_grad = div_const_op(1, node.inputs[1])
        divisor_grad = opposite_op(div_op(div_op(node.inputs[0], node.inputs[1]), node.inputs[1]))
        return [dividend_grad * output_grad, divisor_grad * output_grad]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        assert input_shapes[0] == input_shapes[1], \
            "can't do elementwise division between variables of different sizes."
        output = input_shapes[0]
        return output


class DivConstOp(Op):
    def __call__(self, const_val, node_A):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        if NAME_RULE == 0:
            new_node.name = "(%s/%s)" % (str(const_val), node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "DivConst"
        else:
            new_node.name = "DivConst" + str(new_node.id)
            new_node.desc = new_node.name + "(%s, %s)" % (str(const_val), node_A.name)
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
            from ..gpu_links import matrix_elementwise_divide_const
            matrix_elementwise_divide_const(
                node.const_attr, input_vals[0], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlMatrixElementwiseDivideByConst']:
                from ..ndarray import numpyasdlarrayhandle
                from ..cpu_links import matrix_elementwise_divide_by_const as cpu_matrix_elementwise_divide_by_const
                matA = numpyasdlarrayhandle(input_vals[0])
                matC = numpyasdlarrayhandle(output_val)
                cpu_matrix_elementwise_divide_by_const(matA, node.const_attr, matC)
            else:
                output_val[:] = node.const_attr / input_vals[0]
        else:
            from ..gpu_links import matrix_elementwise_divide_const
            matrix_elementwise_divide_const(
                node.const_attr, input_vals[0], output_val, stream_handle, None)

    def gradient(self, node, output_grad):
        divisor_grad = div_op(div_const_op(-node.const_attr, node.inputs[0]), node.inputs[0])
        return [divisor_grad * output_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


def div_op(node_A, node_B):
    """Make a new instance of matrixs elementwise division and call the instance.

    Parameters:
    ----
    node_A : Node
        The Node where elements are numerators.
    node_B : Node
        Another Node where elements are denominators.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DivOp()(node_A, node_B)


def div_const_op(const_val, node_A):
    """Make a new instance of matrix elementwise devide a constant value and call the instance.

    Parameters:
    ----
    const_val: scalar value
        The constant value to be mutiplied.
    node_A : Node
        The Node where elements are denominators.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return DivConstOp()(const_val, node_A)
