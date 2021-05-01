from __future__ import absolute_import
import numpy as np
import scipy.sparse
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .. import ndarray
from .._base import get_array_memory

class CsrmvOp(Op):
    def __call__(self, node_A, node_B, trans=False):
        new_node = Op.__call__(self)
        new_node.csrmv_attr_trans = trans
        new_node.inputs = [node_A, node_B]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE==0:
          new_node.name = "CsrMatVec(%s,%s,%s)" % (
              node_A.name, node_B.name, str(trans))
        elif NAME_RULE==1:
          new_node.name = "CsrMatVec"
        else:
          new_node.name = "CsrMatVec"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, node_B.name)
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
            assert isinstance(input_vals[0], ndarray.ND_Sparse_Array)
            from ..gpu_links import CuSparse_Csrmv
            CuSparse_Csrmv(
                input_vals[0], node.csrmv_attr_trans,
                input_vals[1], output_val, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True):

        
        assert len(input_vals) == 2
        if use_numpy:
            assert isinstance(input_vals[0], scipy.sparse.spmatrix)
            if node.csrmv_attr_trans is False:
                output_val[:] = input_vals[0].dot(input_vals[1])
            else:
                output_val[:] = input_vals[0].T.dot(input_vals[1])
        else:
            assert isinstance(input_vals[0], ndarray.ND_Sparse_Array)
            from ..gpu_links import CuSparse_Csrmv
            CuSparse_Csrmv(
                input_vals[0], node.csrmv_attr_trans,
                input_vals[1], output_val, None)

    # ND_Sparse_Array gradient not implemented
    def gradient(self, node, output_grad):
        if node.csrmv_attr_trans is False:
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            # lhs_grad = matmul_op(
            #     output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = csrmv_op(
                node.inputs[0], output_grad, trans=True)
        else:
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            # lhs_grad = matmul_op(
            #     node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = csrmv_op(
                node.inputs[0], output_grad, trans=False)
        return [None, rhs_grad]
    
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        A = input_shapes[0]
        B = input_shapes[1]
        assert len(A) == 2 and len(B) == 1
        shape_A = A[0]
        shape_mid_1 = A[1]
        shape_mid_2 = B[0]
        if node.csrmv_attr_trans == True:
            shape_A = A[1]
            shape_mid_1 = A[0]
        assert shape_mid_1 == shape_mid_2
        return (shape_A, )


class CsrmmOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.csrmm_attr_trans_A = trans_A
        new_node.csrmm_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE==0:
          new_node.name = "CsrMatMat(%s,%s,%s,%s)" % (
              node_A.name, node_B.name, str(trans_A), str(trans_B))
        elif NAME_RULE==1:
          new_node.name = "CsrMatMat"
        else:
          new_node.name = "CsrMatMat"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, node_B.name)
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
            assert isinstance(input_vals[0], ndarray.ND_Sparse_Array)
            from ..gpu_links import CuSparse_Csrmm            
            CuSparse_Csrmm(
                input_vals[0], node.csrmm_attr_trans_A,
                input_vals[1], node.csrmm_attr_trans_B,
                output_val, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True):

        assert len(input_vals) == 2
        if use_numpy:
            assert isinstance(input_vals[0], scipy.sparse.spmatrix)
            if ((node.csrmm_attr_trans_A is False) and
                    (node.csrmm_attr_trans_B is False)):
                output_val[:] = input_vals[0].dot(input_vals[1])
            elif ((node.csrmm_attr_trans_A is True) and
                    (node.csrmm_attr_trans_B is False)):
                output_val[:] = input_vals[0].T.dot(input_vals[1])
            elif ((node.csrmm_attr_trans_A is False) and
                    (node.csrmm_attr_trans_B is True)):
                output_val[:] = input_vals[0].dot(np.transpose(input_vals[1]))
            elif ((node.csrmm_attr_trans_A is True) and
                    (node.csrmm_attr_trans_B is True)):
                output_val[:] = input_vals[0].T.dot(np.transpose(input_vals[1]))
        else:
            assert isinstance(input_vals[0], ndarray.ND_Sparse_Array)
            from ..gpu_links import CuSparse_Csrmm            
            CuSparse_Csrmm(
                input_vals[0], node.csrmm_attr_trans_A,
                input_vals[1], node.csrmm_attr_trans_B,
                output_val, None)

    # ND_Sparse_Array gradient not implemented
    def gradient(self, node, output_grad):
        from .Transpose import transpose_op
        if ((node.csrmm_attr_trans_A is False) and
                (node.csrmm_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            # lhs_grad = matmul_op(
            #     output_grad, node.inputs[1], trans_A=False, trans_B=True)
            # Notice: cuSparse not support left trans right not trans
            rhs_grad = csrmm_op(
                node.inputs[0], transpose_op(output_grad), trans_A=True, trans_B=True)
        elif ((node.csrmm_attr_trans_A is True) and
                (node.csrmm_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            # lhs_grad = matmul_op(
            #     node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = csrmm_op(
                node.inputs[0], output_grad, trans_A=False, trans_B=False)
        elif ((node.csrmm_attr_trans_A is False) and
                (node.csrmm_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            # lhs_grad = matmul_op(
            #     output_grad, node.inputs[1], trans_A=False, trans_B=False)
            # rhs_grad = matmul_op(
            #     output_grad, node.inputs[0], trans_A=True, trans_B=False)
            # Notice: cuSparse not support left trans right not trans
            rhs_grad = transpose_op(csrmm_op(
                node.inputs[0], transpose_op(output_grad), trans_A=True, trans_B=True))
        elif ((node.csrmm_attr_trans_A is True) and
                (node.csrmm_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B)^T=B^T dY^T, dB=(A dY)^T=dY^T A^T
            # lhs_grad = matmul_op(
            #     node.inputs[1], output_grad, trans_A=True, trans_B=True)
            # rhs_grad = matmul_op(
            #     output_grad, node.inputs[0], trans_A=True, trans_B=True)
            rhs_grad = transpose_op(csrmm_op(
                node.inputs[0], output_grad, trans_A=False, trans_B=False))
        # return [lhs_grad, rhs_grad]
        return [None, rhs_grad]
    
    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        A = input_shapes[0]
        B = input_shapes[1]
        assert len(A) == 2 and len(B) == 2
        shape_A = A[0]
        shape_B = B[1]
        shape_mid_1 = A[1]
        shape_mid_2 = B[0]
        if node.csrmm_attr_trans_A == True:
            shape_A = A[1]
            shape_mid_1 = A[0]
        if node.csrmm_attr_trans_B == True:
            shape_B = B[0]
            shape_mid_2 = B[1]
        assert shape_mid_1 == shape_mid_2
        return (shape_A, shape_B)     


def csrmv_op(node_A, node_B, trans=False):
    """Make a new instance of multiplication of a sparse matrix and a vector, 
        and call the instance.

    Parameters:
    ----
    node_A : Node
        The left operand, a sparse matrix.
    node_B : Node
        The right operand, a vector.
    trans : Boolean 
        Whether node_A to be transposed, default to be False.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return CsrmvOp()(node_A, node_B, trans)


def csrmm_op(node_A, node_B, trans_A=False, trans_B=False):
    """Make a new instance of Sparse Matrix Multiplication and call the instance.

    Parameters:
    ----
    node_A : Node
        The left operand, a sparse matrix.
    node_B : Node
        The right operand, a dense matrix.
    trans_A : Boolean 
        Whether node_A to be transposed, default to be False.
    trans_B : Boolean 
        Whether node_B to be transposed, default to be False.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return CsrmmOp()(node_A, node_B, trans_A, trans_B)
    