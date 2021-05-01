from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler

class BatchMatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()

        if NAME_RULE == 0:
            new_node.name = "BatchMatMul(%s,%s,%s,%s)" % (
                node_A.name, node_B.name, str(trans_A), str(trans_B))
        elif NAME_RULE == 1:
            new_node.name = "BatchMatMul"
        else:
            new_node.name = "BatchMatMul"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def profile(self, node, input_vals, output_val, is_static=True):
        
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
            from ..gpu_links import batch_matrix_multiply
            batch_matrix_multiply(
                input_vals[0], node.matmul_attr_trans_A,
                input_vals[1], node.matmul_attr_trans_B,
                output_val, None)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        if use_numpy:
            ndims = len(input_vals[0])
            perm = list(range(ndims-2)) + [ndims-1, ndims-2]

            if ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(input_vals[0].asnumpy(), input_vals[1].asnumpy())
            elif ((node.matmul_attr_trans_A is True) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0].asnumpy(), perm), input_vals[1].asnumpy())
            elif ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    input_vals[0].asnumpy(), np.transpose(input_vals[1].asnumpy(), perm))
            elif ((node.matmul_attr_trans_A is True) and
                    (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0].asnumpy(), perm), np.transpose(input_vals[1].asnumpy(), perm))
        else:
            from ..gpu_links import batch_matrix_multiply
            # pass
            batch_matrix_multiply(
                input_vals[0], node.matmul_attr_trans_A,
                input_vals[1], node.matmul_attr_trans_B,
                output_val, stream_handle)

    def gradient(self, node, output_grad):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = batch_matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = batch_matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            lhs_grad = batch_matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = batch_matmul_op(
                node.inputs[0], output_grad, trans_A=False, trans_B=False)
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            lhs_grad = batch_matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=False)
            rhs_grad = batch_matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B)^T=B^T dY^T, dB=(A dY)^T=dY^T A^T
            lhs_grad = batch_matmul_op(
                node.inputs[1], output_grad, trans_A=True, trans_B=True)
            rhs_grad = batch_matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=True)
        return [lhs_grad, rhs_grad]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        A = input_shapes[0]
        B = input_shapes[1]
        assert len(A) == len(B)
        assert len(A) >= 2
        for i in range(len(A)-2):
            assert A[i] == B[i]
        C = list(A)[:-2]
        shape_A = A[-2]
        shape_B = B[-1]
        k1 = A[-1]
        k2 = B[-2]
        if node.matmul_attr_trans_A == True:
            shape_A = A[-1]
            k1 = A[-2]
        if node.matmul_attr_trans_B == True:
            shape_B = B[-2]
            k2 = B[-1]
        assert k1 == k2
        C.extend([shape_A, shape_B])
        # print("batch matmul:", A, B, C)
        return tuple(C)


def batch_matmul_op(node_A, node_B, trans_A=False, trans_B=False):
    """Make a new instance of Batch Matrix Multiplication and call the instance.
    Parameters:
    ----
    node_A : Node
        The left operand of the matrix multiplication.
    node_B : Node
        The right operand of the matrix multiplication.
    trans_A : Boolean 
        Whether node_A to be transposed
    trans_B : Boolean 
        Whether node_B to be transposed
    Returns:
    ----
    A new Node instance created by Op.
    """
    return BatchMatMulOp()(node_A, node_B, trans_A, trans_B)
