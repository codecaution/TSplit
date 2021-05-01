from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "Add"
        else:
            new_node.name = "Add"+str(new_node.id)
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
            from ..gpu_links import matrix_elementwise_add, matrix_elementwise_add_by_const
            if input_vals[0].shape == input_vals[1].shape:
                matrix_elementwise_add(
                    input_vals[0], input_vals[1], output_val, None, node.profiler)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    matrix_elementwise_add_by_const(
                        input_vals[0], const_val, output_val, None, node.profiler)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    matrix_elementwise_add_by_const(
                        input_vals[1], const_val, output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 2
        if use_numpy:
            from ..ndarray import numpyasdlarrayhandle
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlMatrixElementwiseAdd'] and input_vals[0].shape == input_vals[1].shape:
                from ..cpu_links import matrix_elementwise_add as cpu_matrix_elementwise_add
                matA = numpyasdlarrayhandle(input_vals[0])
                matB = numpyasdlarrayhandle(input_vals[1])
                matC = numpyasdlarrayhandle(output_val)
                cpu_matrix_elementwise_add(matA, matB, matC)
            elif DNNL_LIB['DnnlMatrixElementwiseAddByConst'] and (input_vals[1].shape == (1,) or input_vals[0].shape == (1,)):
                from ..cpu_links import matrix_elementwise_add_by_const as cpu_matrix_elementwise_add_by_const
                matC = numpyasdlarrayhandle(output_val)
                if input_vals[1].shape == (1,):
                    matA = numpyasdlarrayhandle(input_vals[0])
                    const_val = input_vals[1][0]
                    cpu_matrix_elementwise_add_by_const(matA, const_val, matC)
                elif input_vals[0].shape == (1,):
                    matB = numpyasdlarrayhandle(input_vals[1])
                    const_val = input_vals[0][0]
                    cpu_matrix_elementwise_add_by_const(matB, const_val, matC)
            else:
                # output_val[:] allows modify in-place
                output_val[:] = input_vals[0] + input_vals[1]
        else:
            from ..gpu_links import matrix_elementwise_add, matrix_elementwise_add_by_const
            if input_vals[0].shape == input_vals[1].shape:
                matrix_elementwise_add(
                    input_vals[0], input_vals[1], output_val, stream_handle, None)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    matrix_elementwise_add_by_const(
                        input_vals[0], const_val, output_val, stream_handle, None)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    matrix_elementwise_add_by_const(
                        input_vals[1], const_val, output_val, stream_handle, None)

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        assert len(input_shapes) == 2

        if input_shapes[0] == input_shapes[1]:
            output = input_shapes[0]
        else:
            if input_shapes[0] == (1,):
                output = input_shapes[1]
            elif input_shapes[1] == (1,):
                output = input_shapes[0]
            else:
                assert False, "can't add variables of different sizes."
        return output

def add_op(node_A, node_B):
    """Make a new instance of Node Addition and call the instance.

    Parameters:
    ----
    node_A : Node
        The Node to be added.
    node_B : Node
        Another Node to be added.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return AddOp()(node_A, node_B)