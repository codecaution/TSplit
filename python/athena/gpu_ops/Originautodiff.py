""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import

import numpy as np
from . import ndarray, gpu_op
import ctypes
import os
from pynvml import *
NAME_RULE = 1
G_NODE_ID = 0
FLAG_SHOW_GRAPH = False

def communicate_init(worker_num, worker_id, source_ip, target_ip): 
    global lib_communicate
    # lib_communicate.DL_Connect_Init(2, 0, "*:4001", "localhost:4002")
    # lib_communicate.DL_Connect_Init(2, 1, "*:4002", "localhost:4001")
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "lib_communication.so")
    lib_communicate = ctypes.cdll.LoadLibrary(path_to_so_file)
    lib_communicate.DL_Connect_Init(worker_num, worker_id, source_ip, target_ip)
def communicate_finish():
    lib_communicate.DL_Communicate_Close()

class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object call method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant.
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
        self.desc = ""
        global G_NODE_ID
        self.id = G_NODE_ID
        G_NODE_ID = G_NODE_ID + 1
        self.swap = False

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Mul by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = mul_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name


def Variable(name, swap = False):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()
    placeholder_node.name = name
    if NAME_RULE == 2:
      placeholder_node.desc = name + str(placeholder_node.id)
    placeholder_node.swap = swap
    return placeholder_node


class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        use_numpy: bool flag whether to use numpy for compute
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given output gradient, compute partial gradient to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """Given shapes of input nodes, compute shape of output node.

        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.

        Parameters
        ----------
        node: node whose shape is being inferred.
        input_vals: shapes of input nodes.

        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError


class AddOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        if NAME_RULE==0:
          new_node.name = "(%s+%s)" % (node_A.name, node_B.name)
        elif NAME_RULE==1:
          new_node.name = "Add"
        else:
          new_node.name = "Add"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            # output_val[:] allows modify in-place
            output_val[:] = input_vals[0] + input_vals[1]
        else:
            if input_vals[0].shape == input_vals[1].shape:
                gpu_op.matrix_elementwise_add(
                    input_vals[0], input_vals[1], output_val)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    gpu_op.matrix_elementwise_add_by_const(
                        input_vals[0], const_val, output_val)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    gpu_op.matrix_elementwise_add_by_const(
                        input_vals[1], const_val, output_val)

    def gradient(self, node, output_grad):
        return [output_grad, output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        # print(input_shapes[0])
        # print(input_shapes[1])
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
        


class AddByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        if NAME_RULE==0:
          new_node.name = "(%s+%s)" % (node_A.name, str(const_val))
        elif NAME_RULE==1:
          new_node.name = "AddbyConst"
        else:
          new_node.name = "AddbyConst"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, str(const_val)) 
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = input_vals[0] + node.const_attr
        else:
            gpu_op.matrix_elementwise_add_by_const(
                input_vals[0], node.const_attr, output_val)

    def gradient(self, node, output_grad):
        return [output_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


class MulOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        if NAME_RULE==0:
          new_node.name = "(%s*%s)" % (node_A.name, node_B.name)
        elif NAME_RULE==1:
          new_node.name = "Mul"
        else:
          new_node.name = "Mul"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = input_vals[0] * input_vals[1]
        else:
            if input_vals[0].shape == input_vals[1].shape:
                gpu_op.matrix_elementwise_multiply(
                    input_vals[0], input_vals[1], output_val)
            else:
                if input_vals[1].shape == (1,):
                    const_val = input_vals[1].asnumpy()[0]
                    gpu_op.matrix_elementwise_multiply_by_const(
                        input_vals[0], const_val, output_val)
                elif input_vals[0].shape == (1,):
                    const_val = input_vals[0].asnumpy()[0]
                    gpu_op.matrix_elementwise_multiply_by_const(
                        input_vals[1], const_val, output_val)

    def gradient(self, node, output_grad):
        return [node.inputs[1] * output_grad, node.inputs[0] * output_grad]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        if input_shapes[0] == input_shapes[1]:
          output = input_shapes[0]
        else:
          if input_shapes[0] == (1,):
            output = input_shapes[1]
          elif input_shapes[1] == (1,):
            output = input_shapes[0]
          else:
            assert False, "can't do elementwise multiply between variables of different sizes."
        return output



class MulByConstOp(Op):
    def __call__(self, node_A, const_val):
        new_node = Op.__call__(self)
        new_node.const_attr = const_val
        new_node.inputs = [node_A]
        if NAME_RULE==0:
          new_node.name = "(%s*%s)" % (node_A.name, str(const_val))
        elif NAME_RULE==1:
          new_node.name = "MulByConst"
        else:
          new_node.name = "MulByConst"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, str(const_val))
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = input_vals[0] * node.const_attr
        else:
            gpu_op.matrix_elementwise_multiply_by_const(
                input_vals[0], node.const_attr, output_val)

    def gradient(self, node, output_grad):
        return [node.const_attr * output_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]



class MatMulOp(Op):
    def __call__(self, node_A, node_B, trans_A=False, trans_B=False):
        new_node = Op.__call__(self)
        new_node.matmul_attr_trans_A = trans_A
        new_node.matmul_attr_trans_B = trans_B
        new_node.inputs = [node_A, node_B]
        if NAME_RULE==0:
          new_node.name = "MatMul(%s,%s,%s,%s)" % (
              node_A.name, node_B.name, str(trans_A), str(trans_B))
        elif NAME_RULE==1:
          new_node.name = "MatMul"
        else:
          new_node.name = "MatMul"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        if use_numpy:
            if ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(input_vals[0], input_vals[1])
            elif ((node.matmul_attr_trans_A is True) and
                    (node.matmul_attr_trans_B is False)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), input_vals[1])
            elif ((node.matmul_attr_trans_A is False) and
                    (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    input_vals[0], np.transpose(input_vals[1]))
            elif ((node.matmul_attr_trans_A is True) and
                    (node.matmul_attr_trans_B is True)):
                output_val[:] = np.matmul(
                    np.transpose(input_vals[0]), np.transpose(input_vals[1]))
        else:
            gpu_op.matrix_multiply(
                input_vals[0], node.matmul_attr_trans_A,
                input_vals[1], node.matmul_attr_trans_B,
                output_val)

    def gradient(self, node, output_grad):
        if ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is False)):
            # if Y=AB, then dA=dY B^T, dB=A^T dY
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is False)):
            # if Y=A^T B, then dA=(dY B^T)^T=B dY^T, dB=A dY
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=False, trans_B=True)
            rhs_grad = matmul_op(
                node.inputs[0], output_grad, trans_A=False, trans_B=False)
        elif ((node.matmul_attr_trans_A is False) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A B^T, then dA=dY B, dB=(A^T dY)^T=dY^T A
            lhs_grad = matmul_op(
                output_grad, node.inputs[1], trans_A=False, trans_B=False)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=False)
        elif ((node.matmul_attr_trans_A is True) and
                (node.matmul_attr_trans_B is True)):
            # if Y=A^T B^T, then dA=(dY B)^T=B^T dY^T, dB=(A dY)^T=dY^T A^T
            lhs_grad = matmul_op(
                node.inputs[1], output_grad, trans_A=True, trans_B=True)
            rhs_grad = matmul_op(
                output_grad, node.inputs[0], trans_A=True, trans_B=True)
        return [lhs_grad, rhs_grad]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        A = input_shapes[0]
        B = input_shapes[1]
        shape_A = A[0]
        shape_B = B[1]
        if node.matmul_attr_trans_A == True: shape_A = A[1]
        if node.matmul_attr_trans_B == True: shape_B = B[0]
        return (shape_A, shape_B)


class PlaceholderOp(Op):
    def __call__(self):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert False, "placeholder %s values provided by feed_dict" % node.name

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes):
        assert False, "placeholder %s shape provided by feed_shape" % node.name


class ZerosLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.zeros(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if NAME_RULE==0:
          new_node.name = "Zeroslike(%s)" % node_A.name
        elif NAME_RULE==1:
          new_node.name = "Zeroslike"
        else:
          new_node.name = "Zeroslike" + str(new_node.id)
          new_node.desc = new_node.name + "(%s)" % new_node.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.zeros(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 0)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


class OnesLikeOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.ones(node_A.shape)."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if NAME_RULE==0:
          new_node.name = "Oneslike(%s)" % node_A.name
        elif NAME_RULE==1:
          new_node.name = "Oneslike"
        else:
          new_node.name = "Oneslike" + str(new_node.id)
          new_node.desc = new_node.name + "(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.ones(input_vals[0].shape)
        else:
            gpu_op.array_set(output_val, 1)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """If input_shape is a vector, simpler to return (1,)"""
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReduceSumAxisZeroOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if NAME_RULE==0:
          new_node.name = "ReduceSumAxisZero(%s)" % (node_A.name)
        elif NAME_RULE==1:
          new_node.name = "ReduceSumAxisZero"
        else:
          new_node.name = "ReduceSumAxisZero" + str(new_node.id)
          new_node.desc = new_node.name + "(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            assert(isinstance(input_vals[0], np.ndarray))
            output_val[:] = np.sum(input_vals[0], axis=0)
        else:
            shapeW = list(input_vals[0].shape)
            shapeW[0] = (shapeW[0] + 1)/2
            shapeW = tuple(shapeW)
            workspace_arr = ndarray.empty(shape = shapeW, ctx = input_vals[0].ctx)
            # workspace_arr = ndarray.empty(shape = input_vals[0].shape, ctx = ndarray.gpu(0))
            gpu_op.reduce_sum_axis_zero(input_vals[0], output_val, workspace_arr)
            # gpu_op._reduce_sum_axis_zero(input_vals[0], output_val, workspace_arr)
            del workspace_arr

    def gradient(self, node, output_grad):
        return [broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        input_shape = input_shapes[0]
        if len(input_shape) == 1:
          return (1,)
        else:
          return input_shape[1:]


class Conv2d_BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape).
        Only support axis=0. e.g. (3,4)->(2,3,4) to make gradient simple.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        if NAME_RULE==0:
            new_node.name = "Conv2d_BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        elif NAME_RULE==1:
            new_node.name = "Conv2d_BroadcastTo"
        else:
            new_node.name = "Conv2d_BroadcastTo" + str(new_node.id)
            new_node.desc = new_node.name + "(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert(len(input_vals)==2)
        # print node.inputs[0].name, node.inputs[1].name
        if use_numpy:
            shapeW = input_vals[1].shape
            shapeW = list(shapeW)
            tmp = shapeW[1]
            shapeW[1] = shapeW[3]
            shapeW[3] = tmp
            output_val[:] = np.broadcast_to(input_vals[0], input_vals[1].shape).swapaxes(1,3)
        else:
            gpu_op.broadcast_to(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        grad_A = conv2d_reducesum_op(output_grad)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        return input_shapes[1]

class Conv2d_ReduceSumOp(Op):
    def __call__(self, node_A):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if NAME_RULE==0:
          new_node.name = "Conv2d_ReduceSum(%s)" % (node_A.name)
        elif NAME_RULE==1:
          new_node.name = "Conv2d_ReduceSum"
        else:
          new_node.name = "ReduceSumAxisZero" + str(new_node.id)
          new_node.desc = new_node.name + "(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            assert(isinstance(input_vals[0], np.ndarray))
            output_val[:] = np.sum(input_vals[0], axis=(0,2,3))
        else:
            gpu_op.conv2d_reduce_sum(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [conv2d_broadcastto_op(output_grad, node.inputs[0])]

    def infer_shape(self, node, input_shapes):
        """summation reduction axis = 0
        e.g. (3,4,5)->(4,5)
        for vector, simpler to do (3,)->(1,)
        """
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        # input_shape = input_shapes[0]
        # if len(input_shape) == 1:
        #   return (1,)
        # else:
        #   return input_shape[1:]
        channels = input_shapes[0][1]
        return (channels,)


class BroadcastToOp(Op):
    def __call__(self, node_A, node_B):
        """Creates a node that represents np.broadcast_to(node_A, node_B.shape).
        Only support axis=0. e.g. (3,4)->(2,3,4) to make gradient simple.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        if NAME_RULE==0:
          new_node.name = "BroadcastTo(%s,%s.shape)" % (node_A.name, node_B.name)
        elif NAME_RULE==1:
          new_node.name = "BroadcastTo"
        else:
          new_node.name = "BroadcastTo" + str(new_node.id)
          new_node.desc = new_node.name + "(%s,%s.shape)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert(len(input_vals)==2)
        # print node.inputs[0].name, node.inputs[1].name
        if use_numpy:
            output_val[:] = np.broadcast_to(input_vals[0], input_vals[1].shape)
        else:
            gpu_op.broadcast_to(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        grad_A = reducesumaxiszero_op(output_grad)
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        return input_shapes[1]

def softmax_func(y):
    """Numerically stable softmax."""
    b = y - np.max(y, axis=1, keepdims=True)
    expb = np.exp(b)
    softmax = expb / np.sum(expb, axis=1, keepdims=True)
    return softmax


class SoftmaxCrossEntropyOp(Op):
    def __call__(self, node_A, node_B):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        if NAME_RULE==0:
          new_node.name = "SoftmaxXEntropy(%s,%s)" % (node_A.name, node_B.name)
        elif NAME_RULE==1:
          new_node.name = "SoftmaxEntropy"
        else:
          new_node.name = "SoftmaxEntropy" + str(new_node.id)
          new_node.desc = new_node.name + "(%s,%s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        y = input_vals[0]
        y_ = input_vals[1]
        if use_numpy:
            softmax = softmax_func(y)
            cross_entropy = np.mean(
                -np.sum(y_ * np.log(softmax), axis=1), keepdims=True)
            output_val[:] = cross_entropy
        else:
            gpu_op.softmax_cross_entropy(y, y_, output_val)

    def gradient(self, node, output_grad):
        grad_A = (softmax_op(node.inputs[0]) + -1 * node.inputs[1])*output_grad
        grad_B = zeroslike_op(node.inputs[1])
        return [grad_A, grad_B]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        return (1,)


class SoftmaxOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if NAME_RULE==0:
          new_node.name = "Softmax(%s)" % (node_A.name)
        elif NAME_RULE==1:
          new_node.name = "Softmax"
        else:
          new_node.name = "Softmax" + str(new_node.id)
          new_node.desc = new_node.name + "(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = softmax_func(input_vals[0])
        else:
            gpu_op.softmax(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        # Do not directly use SoftmaxOp, use SoftmaxCrossEntropyOp instead.
        # Not allowing taking 2nd derivative of SoftmaxCrossEntropyOp.
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluOp(Op):
    def __call__(self, node_A):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        if NAME_RULE == 0:
          new_node.name = "Relu(%s)" % (node_A.name)
        elif NAME_RULE == 1:
          new_node.name = "Relu"
        else:
          new_node.name = "Relu" + str(new_node.id)
          new_node.desc = new_node.name + "(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            output_val[:] = np.maximum(input_vals[0], 0)
        else:
            gpu_op.relu(input_vals[0], output_val)

    def gradient(self, node, output_grad):
        return [relu_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        return input_shapes[0]


class ReluGradientOp(Op):
    def __call__(self, node_A, node_B):
        """node_B is output_grad"""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        if NAME_RULE == 0:
          new_node.name = "ReluGradient(%s)" % (node_A.name)
        elif NAME_RULE == 1:
          new_node.name = "ReluGradient"
        else:
          new_node.name = "ReluGradient" + str(new_node.id)
          new_node.desc = new_node.name + "(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            # heaviside function, 0.5 at x=0
            output_val[:] = (np.sign(input_vals[0]) + 1) * 0.5 * input_vals[1]
        else:
            gpu_op.relu_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        return input_shapes[0]

class Conv2dOp(Op):
    # nodeA : x  nodeB : filter
    def __call__(self, node_A, node_B,padding = 0, stride = 1):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        
        # self.padding_H = self.padding_W = 0
        # self.stride_H = self.stride_W = 0
        # if(isinstance(padding, int)):
        #   self.padding_H = self.padding_W = padding
        # else if (len( padding) == 1):
        #   self.padding_H = self.padding_W = padding[0]
        # else:
        #   assert len(padding) == 2
        #   self.padding_H = padding[0]
        #   self.padding_W = padding[1]
        
        # if(isinstance(stride, int)):
        #   self.stride_H = self.stride_W = stride
        # else if(len(stride) == 1):
        #   self.stride_H = self.stride_W = stride[0]
        # else:
        #   assert len(stride) == 2
        #   self.stride_H = stride[0]
        #   self.stride_W = stride[1]
        self.padding = padding
        self.stride = stride
        
        # print "init padding = ", padding
        if NAME_RULE == 0:
          new_node.name = "Conv2d(%s, %s)" % (node_A.name, node_B.name)
        elif NAME_RULE == 1:
          new_node.name = "Conv2d"
        else:
          new_node.name = "conv2d"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def im2col(self, X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype = X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index / out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx, col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y
        
    def np_conv2d(self, X, Filter, padding = 0, stride = 1):
        """Implement a conv2d as a matrix multiply after im2col."""
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1

        im2col_matrix = self.im2col(X, filter_H, filter_W, padding, stride)
        filter_matrix = Filter.reshape(filter_outChannel, -1)
        return np.matmul(filter_matrix, im2col_matrix).reshape(N, filter_outChannel, out_H, out_W)

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            output_val[:] = self.np_conv2d(input_vals[0], input_vals[1], self.padding, self.stride)
        else:
            # N, C, H, W = input_vals[0].shape
            # _, filter_C, filter_H, filter_W = input_vals[1].shape
            # padding = self.padding
            # stride = self.stride
            # out_H = (H + 2 * padding - filter_H) / stride + 1
            # out_W = (W + 2 * padding - filter_W) / stride + 1
            # shapeW = (N, filter_C * filter_H * filter_W, out_H * out_W)
            # arr_workspace = ndarray.empty(shapeW, ctx = ndarray.gpu(0))
            # gpu_op.conv2d(input_vals[0], input_vals[1], output_val, arr_workspace, self.padding, self.stride) 
            # del arr_workspace
            # print(self.stride, self.padding)
            gpu_op.CuDNN_conv2d(input_vals[0], input_vals[1], output_val,self.padding, self.stride) 

    def gradient(self, node, output_grad):
        return [conv2d_gradient_of_data_op(node.inputs[1], output_grad, self.padding, self.stride), conv2d_gradient_of_filter_op(node.inputs[0], output_grad, self.padding, self.stride)]
    

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        # print "infer padding = ",self.padding
        N, _, H, W = input_shapes[0]
        f_O, _, f_H, f_W = input_shapes[1]
        padding = self.padding
        stride = self.stride
        filter_H = input_shapes[1][2]
        filter_W = input_shapes[1][3]
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1
        # print "conv2d_shape"
        # print(N, f_O, out_H, out_W)
        return (N, f_O, out_H, out_W) 

class Conv2dOp_call():
    def __call__(self, node_A, node_B,padding = 0, stride = 1):
        new_op = Conv2dOp()
        return new_op(node_A, node_B, padding, stride)

class Conv2d_Gradient_of_DataOp(Op):
    # nodeA : filter  nodeB : Y_gradient 
    def __call__(self, node_A, node_B,padding = 0, stride = 1):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        
        self.padding = padding
        self.stride = stride

        if NAME_RULE == 0:
          new_node.name = "Conv2d_Gradient_of_DataOp(%s, %s)" % (node_A.name, node_B.name)
        elif NAME_RULE == 1:
          new_node.name = "Conv2d_Gradient_of_DataOp"
        else:
          new_node.name = "Conv2d_Gradient_of_DataOp"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_A.name, node_B.name)
        return new_node
    def im2col_transpose(self, N, C, H, W, filter_H, filter_W, Y , padding, stride):
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1
        _, y_row_size, y_col_size = Y.shape

        der_X_shape = (N, C, H, W)
        der_X = np.zeros(der_X_shape, dtype = Y.dtype)

        # print "batch_size", N
        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index / out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                der_X[batch_index, c, y, x] += Y[batch_index, row_idx, col_index] 
                            row_idx += 1
        return der_X

    def np_Conv2dGradient_data(self, X_N, X_C, X_H, X_W, Filter, Y, padding = 0, stride = 1):
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        F_filter = Filter.reshape((filter_outChannel,-1))

        gradient_im2col_XX = np.matmul(F_filter.T, YY) 
        gradient_X = self.im2col_transpose(X_N, X_C, X_H, X_W, filter_H, filter_W, gradient_im2col_XX, padding, stride)    # gradient of x
        return gradient_X
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        N = input_vals[1].shape[0]
        C = input_vals[0].shape[1]
        H = (input_vals[1].shape[2] - 1) * self.stride + input_vals[0].shape[2] - 2 * self.padding
        W = (input_vals[1].shape[3] - 1) * self.stride + input_vals[0].shape[3] - 2 * self.padding
        if use_numpy:
            output_val[:] = self.np_Conv2dGradient_data(N, C, H, W,input_vals[0], input_vals[1], padding = self.padding, stride = self.stride)
        else:
            # _, filter_C, filter_H, filter_W = input_vals[0].shape
            # N, _, out_H, out_W = input_vals[1].shape
            # shapeW = (N, filter_C * filter_H * filter_W, out_H * out_W)
            # arr_workspace_im2col = ndarray.empty(shapeW, ctx = ndarray.gpu(0))
            # gpu_op.conv2d_gradient_of_data(input_vals[0], input_vals[1], output_val, arr_workspace_im2col, padding = self.padding, stride = self.stride) 
            # del arr_workspace_im2col
            gpu_op.CuDNN_conv2d_gradient_of_data(input_vals[0], input_vals[1], output_val, padding = self.padding, stride = self.stride) 

    def gradient(self, node, output_grad):
        raise NotImplementedError 
    

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        N = input_shapes[1][0]
        C = input_shapes[0][1]
        H = (input_shapes[1][2] - 1) * self.stride + input_shapes[0][2] - 2 * self.padding
        W = (input_shapes[1][3] - 1) * self.stride + input_shapes[0][3] - 2 * self.padding
        return (N, C, H, W) 

class Conv2d_Gradient_of_DataOp_call(): 
    def __call__(self, node_A, node_B, padding = 0, stride = 1):
        new_op = Conv2d_Gradient_of_DataOp()
        return new_op(node_A, node_B, padding, stride)

class Conv2d_Gradient_of_FilterOp(Op):
    # nodeA : input_x  nodeB : gradient_Y
    def __call__(self, input_X, gradient_Y,padding = 0, stride = 1):
        new_node = Op.__call__(self)
        new_node.inputs = [input_X, gradient_Y]
        
        self.padding = padding
        self.stride = stride

        if NAME_RULE == 0:
          new_node.name = "Conv2d_Gradient_of_FilterOp(%s, %s)" % (input_X.name, gradient_Y.name)
        elif NAME_RULE == 1:
          new_node.name = "Conv2d_Gradient_of_FilterOp"
        else:
          new_node.name = "Conv2d_Gradient_of_FilterOp"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (input_X.name, gradient_Y.name)
        return new_node
    def im2col(self, X, filter_H, filter_W, padding, stride):
        N, C, H, W = X.shape
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1

        y_row_size = C * filter_H * filter_W
        y_col_size = out_H * out_W
        y_shape = (N, y_row_size, y_col_size)
        Y = np.empty(y_shape, dtype = X.dtype)

        for batch_index in range(N):
            for col_index in range(y_col_size):
                out_y = col_index / out_W
                out_x = col_index % out_W
                in_y = out_y * stride - padding
                in_x = out_x * stride - padding
                row_idx = 0
                for c in range(0, C):
                    for y in range(in_y, in_y + filter_H):
                        for x in range(in_x, in_x + filter_W):
                            if (x < 0 or x >= W or y < 0 or y >= H):
                                Y[batch_index, row_idx, col_index] = 0
                            else:
                                Y[batch_index, row_idx, col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y
    def np_Conv2dGradient_Filter(self ,filter_outChannel, filter_inChannel, filter_H, filter_W, X, Y, padding = 0, stride = 1):
        """Implement a conv2d_transpose as a matrix multiply after im2col."""
        X_N, X_C, X_H, X_W = X.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        # XX = X.reshape((X_N, X_C, X_W * X_H))   # transformed to im2col X
        im2col_XX = self.im2col(X, filter_H, filter_W, padding, stride)
        gradient_filter = np.zeros(shape = (filter_outChannel, filter_inChannel * filter_H * filter_W), dtype = Y.dtype)

        for i in range(X_N):
          gradient_filter += np.matmul(YY[i],im2col_XX[i].T)
        gradient_filter = gradient_filter.reshape((filter_outChannel, filter_inChannel, filter_H, filter_W))

        return gradient_filter
        # out_H = (H + 2 * padding - filter_H) / stride + 1
        # out_W = (W + 2 * padding - filter_W) / stride + 1
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        f_N = input_vals[1].shape[1]
        f_C = input_vals[0].shape[1]
        f_H = input_vals[1].shape[2] + 2 * self.padding - (input_vals[1].shape[2] - 1) * self.stride
        f_W = input_vals[1].shape[3] + 2 * self.padding - (input_vals[1].shape[3] - 1) * self.stride
        if use_numpy:
            output_val[:] = self.np_Conv2dGradient_Filter(f_N, f_C, f_H, f_W,input_vals[0], input_vals[1], padding = self.padding, stride = self.stride)
        else:
            # N, _, out_H, out_W = input_vals[1].shape
            # shapeW = (N, f_C * f_H * f_W, out_H * out_W)
            # shapeF = (N, f_N, f_C, f_H, f_W)
            # arr_workspace_batch_filter = ndarray.empty(shapeF, ctx = ndarray.gpu(0))
            # arr_workspace_im2col = ndarray.empty(shapeW, ctx = ndarray.gpu(0))
            # gpu_op.conv2d_gradient_of_filter(input_vals[0], input_vals[1], output_val, arr_workspace_im2col, arr_workspace_batch_filter ,padding = self.padding, stride = self.stride) 
            # del arr_workspace_im2col
            # del arr_workspace_batch_filter
            gpu_op.CuDNN_conv2d_gradient_of_filter(input_vals[0], input_vals[1], output_val,padding = self.padding, stride = self.stride) 
 

    def gradient(self, node, output_grad):
        raise NotImplementedError 
    

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        f_N = input_shapes[1][1]
        f_C = input_shapes[0][1]
        f_H = input_shapes[0][2] + 2 * self.padding - (input_shapes[1][2] - 1) * self.stride
        f_W = input_shapes[0][3] + 2 * self.padding - (input_shapes[1][3] - 1) * self.stride
        
        return (f_N, f_C, f_H, f_W) 

class Conv2d_Gradient_of_FilterOp_call():
    def __call__(self, input_X, gradient_Y,padding = 0, stride = 1):
        new_op = Conv2d_Gradient_of_FilterOp()
        return new_op(input_X, gradient_Y, padding, stride)

def np_max_pooling(input, kernel_H, kernel_W, padding = 0, stride = 1):
    N, C, H, W = input.shape
    assert((H + 2 * padding - kernel_H)%stride == 0)
    assert((W + 2 * padding - kernel_W)%stride == 0)
    pooled_H = (H + 2 * padding - kernel_H) / stride + 1
    pooled_W = (W + 2 * padding - kernel_W) / stride + 1

    pooled_layer = np.zeros(shape = ( N , C, pooled_H, pooled_W), dtype = np.float32)
    pooling_size = kernel_H * kernel_W

    for n in range(N):
        for c in range(C):
            for h in range(pooled_H):
                for w in range(pooled_W):
                    hs = h * stride - padding
                    ws = w * stride - padding
                    hend = min(hs + kernel_H, H)
                    wend = min(ws + kernel_W, W)
                    hs = max(hs, 0)
                    ws = max(ws, 0)

                    hargmax = hs
                    wargmax = ws
                    for i in range(hs, hend):
                        for j in range(ws, wend):
                            if input[n][c][i][j] > input[n][c][hargmax][wargmax]:
                                hargmax = i
                                wargmax = j
                    pooled_layer[n][c][h][w] = input[n][c][hargmax][wargmax]

    return pooled_layer

# class Max_Pool2dOp(Op):
#     def __call__(self, node_A, kernel_H, kernel_W, padding, stride):
#         new_node = Op.__call__(self)
#         new_node.inputs = [node_A]
#         self.padding = padding
#         self.stride = stride
#         self.kernel_H = kernel_H
#         self.kernel_W = kernel_W

#         if NAME_RULE==0:
#           new_node.name = "(%s)" % (node_A.name)
#         elif NAME_RULE==1:
#           new_node.name = "Max_Pool2d"
#         else:
#           new_node.name = "Max_Pool2d"+str(new_node.id)
#           new_node.desc = new_node.name+"(%s)" % (node_A.name)
#         return new_node

#     def compute(self, node, input_vals, output_val, use_numpy=True):
#         assert len(input_vals) == 1
#         if use_numpy:
#             # output_val[:] allows modify in-place
#             output_val[:] = np_max_pooling(input_vals[0], self.kernel_H, self.kernel_W, self.padding, self.stride)
#         else:
#             N1, C1, H1, W1 = input_vals[0].shape
#             N2, C2, H2, W2 = output_val.shape
#             assert(N1 == N2)
#             assert(C1 == C2)
#             assert((H1 + 2 * self.padding - self.kernel_H) / self.stride + 1 == H2)
#             assert((W1 + 2 * self.padding - self.kernel_W) / self.stride + 1 == W2)
#             gpu_op.max_pooling2d(input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride) 

#     def gradient(self, node, output_grad):
#         return [max_pool2d_gradient_op(node.inputs[0], output_grad, self.kernel_H, self.kernel_W, self.padding, self.stride)]

#     def infer_shape(self, node, input_shapes):
#         """Need to handle input_vals[0].shape != input_vals[1].shape"""
#         """TODO: Your code here"""
#         assert len(input_shapes) == 1
#         N,C,H,W = input_shapes[0]
#         p_H = (H + 2 * self.padding - self.kernel_H) / self.stride + 1
#         p_W = (W + 2 * self.padding - self.kernel_W) / self.stride + 1
#         return (N, C, p_H, p_W)


# def np_max_pooling_gradient(input, gradient_y, kernel_H, kernel_W, padding = 0, stride = 1):
#     N, C , pooled_H, pooled_W = gradient_y.shape
#     H = (pooled_H - 1) * stride + kernel_H - 2 * padding
#     W = (pooled_W - 1) * stride + kernel_W - 2 * padding

#     gradient_x = np.zeros(shape = (N, C, H, W), dtype = np.float32)
#     pooling_size = kernel_H * kernel_W

#     for n in range(N):
#         for c in range(C):
#             for h in range(pooled_H):
#                 for w in range(pooled_W):
#                     hs = h * stride - padding
#                     ws = w * stride - padding
#                     hend = min(hs + kernel_H, H) 
#                     wend = min(ws + kernel_W, W)
#                     hs = max(hs, 0)
#                     ws = max(ws, 0)

#                     hargmax = hs
#                     wargmax = ws
#                     for i in range(hs, hend):
#                         for j in range(ws, wend):
#                             if input[n][c][i][j] > input[n][c][hargmax][wargmax]:
#                                 hargmax = i
#                                 wargmax = j
#                     gradient_x[n][c][hargmax][wargmax] += gradient_y[n][c][h][w]

#     return gradient_x

# class Max_Pool2d_GradientOp(Op):
#     def __call__(self, node_A, node_B, kernel_H, kernel_W, padding, stride):
#         new_node = Op.__call__(self)
#         new_node.inputs = [node_A, node_B]
#         self.padding = padding
#         self.stride = stride
#         self.kernel_H = kernel_H
#         self.kernel_W = kernel_W

#         if NAME_RULE==0:
#           new_node.name = "(%s,%s)" % (node_A.name, node_B.name)
#         elif NAME_RULE==1:
#           new_node.name = "Max_Pool2d_Gradient"
#         else:
#           new_node.name = "Max_Pool2d_Gradient"+str(new_node.id)
#           new_node.desc = new_node.name+"(%s,%s)" % (node_A.name, node_B.name)
#         return new_node

#     def compute(self, node, input_vals, output_val, use_numpy=True):
#         assert len(input_vals) == 2
#         if use_numpy:
#             # output_val[:] allows modify in-place
#             output_val[:] = np_max_pooling_gradient(input_vals[0], input_vals[1], self.kernel_H, self.kernel_W, self.padding, self.stride)
#         else:
#             N1, C1, H1, W1 = input_vals[0].shape
#             N2, C2, H2, W2 = input_vals[1].shape
#             N3, C3, H3, W3 = output_val.shape
#             assert((H1 + 2 * self.padding - self.kernel_H) / self.stride + 1 == H2)
#             assert((W1 + 2 * self.padding - self.kernel_W) / self.stride + 1 == W2)

#             gpu_op.max_pooling2d_gradient(input_vals[0], input_vals[1], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride) 

#     def gradient(self, node, output_grad):
#         NotImplementedError

#     def infer_shape(self, node, input_shapes):
#         """Need to handle input_vals[0].shape != input_vals[1].shape"""
#         """TODO: Your code here"""
#         assert len(input_shapes) == 2
#         N,C,H,W = input_shapes[0]
#         p_H = (H - 1) * self.stride - 2 * self.padding + self.kernel_H
#         p_W = (W - 1) * self.stride - 2 * self.padding + self.kernel_W
#         return (N, C, p_H, p_W)

class Max_Pool2dOp(Op):
    def __call__(self, node_A, kernel_H, kernel_W, padding, stride):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W

        if NAME_RULE==0:
          new_node.name = "(%s)" % (node_A.name)
        elif NAME_RULE==1:
          new_node.name = "Max_Pool2d"
        else:
          new_node.name = "Max_Pool2d"+str(new_node.id)
          new_node.desc = new_node.name+"(%s)" % (node_A.name)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            # output_val[:] allows modify in-place
            # output_val[:] = self.np_average_pooling(input_vals[0], self.kernel_H, self.kernel_W, self.padding, self.stride)
            raise NotImplementedError
        else:
            N, C, H, W = input_vals[0].shape
            _N, _C, _H, _W = output_val.shape
            assert(N == _N)
            assert(C == _C)
            assert((H + 2 * self.padding - self.kernel_H) / self.stride + 1 == _H)
            assert((W + 2 * self.padding - self.kernel_W) / self.stride + 1 == _W)
            # gpu_op.average_pooling2d(input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride) 
            gpu_op.CuDNN_max_pooling2d(input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride) 

    def gradient(self, node, output_grad):
        return [max_pool2d_gradient_op(node, output_grad, node.inputs[0], self.kernel_H, self.kernel_W, self.padding, self.stride)]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        N,C,H,W = input_shapes[0]
        p_H = (H + 2 * self.padding - self.kernel_H) / self.stride + 1
        p_W = (W + 2 * self.padding - self.kernel_W) / self.stride + 1
        return (N, C, p_H, p_W)


class Max_Pool2d_GradientOp(Op):
    def __call__(self, node_out,node_out_gradient, node_in, kernel_H, kernel_W, padding, stride):
        new_node = Op.__call__(self)
        new_node.inputs = [node_out, node_out_gradient, node_in]
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W

        if NAME_RULE==0:
          new_node.name = "(%s, %s, %s)" % (node_out.name, node_out_gradient.name, node_in.name)
        elif NAME_RULE==1:
          new_node.name = "Max_Pool2d_Gradient"
        else:
          new_node.name = "Max_Pool2d_Gradient"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s, %s)" % (node_out.name, node_out_gradient.name, node_in.name)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        if use_numpy:
            # output_val[:] allows modify in-place
            # output_val[:] = self.np_average_pooling_gradient(input_vals[0], self.kernel_H, self.kernel_W, self.padding, self.stride)
            raise NotImplementedError
        else:
            N, C, H, W = input_vals[0].shape
            _N, _C, _H, _W = output_val.shape
            assert(N == _N)
            assert(C == _C)
            assert((_H + 2 * self.padding - self.kernel_H) / self.stride + 1 == H)
            assert((_W + 2 * self.padding - self.kernel_W) / self.stride + 1 == W)
            gpu_op.CuDNN_max_pooling2d_gradient(input_vals[0], input_vals[1], input_vals[2], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride) 

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[2]

class Avg_Pool2dOp(Op):
    def __call__(self, node_A, kernel_H, kernel_W, padding, stride):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W

        if NAME_RULE==0:
          new_node.name = "(%s)" % (node_A.name)
        elif NAME_RULE==1:
          new_node.name = "Avg_Pool2d"
        else:
          new_node.name = "Avg_Pool2d"+str(new_node.id)
          new_node.desc = new_node.name+"(%s)" % (node_A.name)
        return new_node
    def np_average_pooling(self, input, kernel_H, kernel_W, padding = 0, stride = 1):
        N, C, H, W = input.shape
        assert((H + 2 * padding - kernel_H)%stride == 0)
        assert((W + 2 * padding - kernel_W)%stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) / stride + 1
        pooled_W = (W + 2 * padding - kernel_W) / stride + 1
        pooled_layer = np.zeros(shape = ( N , C, pooled_H, pooled_W), dtype = np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H)
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                pooled_layer[n][c][h][w] += input[n][c][i][j]
                        pooled_layer[n][c][h][w] /= pooling_size
        return pooled_layer
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            # output_val[:] allows modify in-place
            output_val[:] = self.np_average_pooling(input_vals[0], self.kernel_H, self.kernel_W, self.padding, self.stride)
        else:
            N, C, H, W = input_vals[0].shape
            _N, _C, _H, _W = output_val.shape
            assert(N == _N)
            assert(C == _C)
            assert((H + 2 * self.padding - self.kernel_H) / self.stride + 1 == _H)
            assert((W + 2 * self.padding - self.kernel_W) / self.stride + 1 == _W)
            # gpu_op.average_pooling2d(input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride) 
            gpu_op.CuDNN_average_pooling2d(input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride) 

    def gradient(self, node, output_grad):
        return [avg_pool2d_gradient_op(node, output_grad, node.inputs[0], self.kernel_H, self.kernel_W, self.padding, self.stride)]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        N,C,H,W = input_shapes[0]
        p_H = (H + 2 * self.padding - self.kernel_H) / self.stride + 1
        p_W = (W + 2 * self.padding - self.kernel_W) / self.stride + 1
        return (N, C, p_H, p_W)


class Avg_Pool2d_GradientOp(Op):
    def __call__(self, node_out,node_out_gradient, node_in, kernel_H, kernel_W, padding, stride):
        new_node = Op.__call__(self)
        new_node.inputs = [node_out, node_out_gradient, node_in]
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W

        if NAME_RULE==0:
          new_node.name = "(%s, %s, %s)" % (node_out.name, node_out_gradient.name, node_in.name)
        elif NAME_RULE==1:
          new_node.name = "Avg_Pool2d_Gradient"
        else:
          new_node.name = "Avg_Pool2d_Gradient"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s, %s)" % (node_out.name, node_out_gradient.name, node_in.name)
        return new_node

    def np_average_pooling_gradient(self, gradient_y, kernel_H, kernel_W, padding = 0, stride = 1):
        N, C , pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding
        
        gradient_x = np.zeros(shape = (N, C, H, W), dtype = np.float32)
        pooling_size = kernel_H * kernel_W
        for n in range(N):
            for c in range(C):
                for h in range(pooled_H):
                    for w in range(pooled_W):
                        hs = h * stride - padding
                        ws = w * stride - padding
                        hend = min(hs + kernel_H, H) 
                        wend = min(ws + kernel_W, W)
                        hs = max(hs, 0)
                        ws = max(ws, 0)
                        for i in range(hs, hend):
                            for j in range(ws, wend):
                                gradient_x[n][c][i][j] += gradient_y[n][c][h][w] / pooling_size

        return gradient_x
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 3
        if use_numpy:
            # output_val[:] allows modify in-place
            output_val[:] = self.np_average_pooling_gradient(input_vals[0], self.kernel_H, self.kernel_W, self.padding, self.stride)
        else:
            N, C, H, W = input_vals[0].shape
            _N, _C, _H, _W = output_val.shape
            assert(N == _N)
            assert(C == _C)
            assert((_H + 2 * self.padding - self.kernel_H) / self.stride + 1 == H)
            assert((_W + 2 * self.padding - self.kernel_W) / self.stride + 1 == W)
            gpu_op.CuDNN_average_pooling2d_gradient(input_vals[0], input_vals[1], input_vals[2], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride) 

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[2]

class Array_ReshapeOp(Op):
    def __call__(self, node_A, output_shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.output_shape = output_shape
        if NAME_RULE==0:
          new_node.name = "(%s)" % (node_A.name)
        elif NAME_RULE==1:
          new_node.name = "Array_Reshape_Op"
        else:
          new_node.name = "Array_Reshape_Op"+str(new_node.id)
          new_node.desc = new_node.name+"(%s)" % (node_A.name)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert( len(input_vals) == 1)
        input_size = 1
        for i in range(len(input_vals[0].shape)):
            input_size *= input_vals[0].shape[i]
        # check if there exists -1 in output_shape
        idx = -1
        cnt = 0
        output_size = 1
        output_shape = list(self.output_shape)
        for i in range(len(output_shape)):
            if(output_shape[i] == -1):
                idx = i
                cnt = cnt + 1
                assert(cnt != 2)
            output_size *=output_shape[i]

        if(idx == -1):
            assert input_size == output_size
        else:
            output_size = output_size * (-1)
            assert (input_size % output_size == 0)
            output_shape[idx] = input_size / output_size
        output_shape = tuple(output_shape)
        self.output_shape = output_shape
        # print "reshape input_shape",self.input_shape
        # print "reshape output_shape ", output_shape
        # print input_vals[0].shape,output_val.shape
        if use_numpy:
            output_val[:] = input_vals[0].reshape(output_shape)
        else:
            gpu_op.array_reshape(input_vals[0], output_val)
    def gradient(self, node, output_grad):
        return [array_reshape_gradient_op(node.inputs[0], output_grad)]

    def infer_shape(self, node, input_shapes):

        assert (len(input_shapes) == 1)
        input_size = 1
        input_shape = input_shapes[0]
        for i in range(len(input_shape)):
            input_size *= input_shape[i]
        
        # check if there exists -1 in output_shape
        idx = -1
        cnt = 0
        output_size = 1
        output_shape = list(self.output_shape)
        for i in range(len(output_shape)):
            if(output_shape[i] == -1):
                idx = i
                cnt = cnt + 1
                assert(cnt != 2)
            output_size *=output_shape[i]
        if(idx == -1):
            assert input_size == output_size
        else:
            output_size = output_size * (-1)
            assert (input_size % output_size == 0)
            output_shape[idx] = input_size / output_size
        output_shape = tuple(output_shape)
        self.output_shape = output_shape
        # print input_shapes[0]
        # print output_shape
        return output_shape            

class Array_Reshape_GradientOp(Op):
    def __call__(self, node_in, node_out):
        new_node = Op.__call__(self)
        new_node.inputs = [node_in,node_out]

        if NAME_RULE==0:
          new_node.name = "(%s,%s)" % (node_in.name, node_out.name)
        elif NAME_RULE==1:
          new_node.name = "Array_Reshape_GradientOp"
        else:
          new_node.name = "Array_Reshape_GradientOp"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_in.name, node_out.name)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy=True):
        # the size of input_array
        shapeIn = input_vals[0].shape
        # print input_vals[1].shape         
        if use_numpy:
            output_val[:] = input_vals[1].reshape(shapeIn)
        else:
            gpu_op.array_reshape(input_vals[1], output_val)
    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        # print input_shapes[0]
        return input_shapes[0] 

class DropoutOp(Op):
    def __call__(self, node_in, keep_prob):
        new_node = Op.__call__(self)
        new_node.inputs = [node_in, keep_prob]

        if NAME_RULE==0:
          new_node.name = "(%s,%s)" % (node_in.name, keep_prob.name)
        elif NAME_RULE==1:
          new_node.name = "DropoutOp"
        else:
          new_node.name = "DropoutOp"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_in.name, keep_prob.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):

        if use_numpy:
            raise NotImplementedError
        else:
            gpu_op.CuDNN_Dropout(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):

        return dropout_gradient_op(output_grad, node.inputs[1]), zeroslike_op(node.inputs[1])

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

class Dropout_GradientOp(Op):
    def __call__(self, node_in, keep_prob):
        new_node = Op.__call__(self)
        new_node.inputs = [node_in, keep_prob]

        if NAME_RULE==0:
          new_node.name = "(%s,%s)" % (node_in.name, keep_prob.name)
        elif NAME_RULE==1:
          new_node.name = "Dropout_GradientOp"
        else:
          new_node.name = "Dropout_GradientOp"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (node_in.name, keep_prob.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):

        if use_numpy:
            raise NotImplementedError
        else:
            gpu_op.CuDNN_Dropout_gradient(input_vals[0], input_vals[1], output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class Batch_NormalizationOp(Op):
    def __call__(self, node_in, bn_scale, bn_bias, momentum = 0.99, eps = 0.01):
        new_node = Op.__call__(self)
        new_node.inputs = [node_in, bn_scale, bn_bias]
        self.momentum = momentum
        self.eps = eps
        if NAME_RULE==0:
          new_node.name = "(%s,%s,%s)" % (node_in.name, bn_scale.name, bn_bias.name)
        elif NAME_RULE==1:
          new_node.name = "Batch_NormalizationOp"
        else:
          new_node.name = "Batch_NormalizationOp"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s, %s)" % (node_in.name, bn_scale.name, bn_bias.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):

        if use_numpy:
            raise NotImplementedError
        else:
            gpu_op.CuDNN_Batch_Normalization(input_vals[0], input_vals[1], input_vals[2], output_val, self.momentum, self.eps)

    def gradient(self, node, output_grad):

        bn_gradient_node = batch_normalization_gradient_op(output_grad, node.inputs[0], node.inputs[1])

        data_gradient = batch_normalization_gradient_of_data_op(bn_gradient_node, node.inputs[0])
        scale_gradient = batch_normalization_gradient_of_scale_op(bn_gradient_node, node.inputs[1])
        bias_gradient = batch_normalization_gradient_of_bias_op(bn_gradient_node, node.inputs[2])
        #data_gradient = batch_normalization_gradient_of_data_op(output_grad, node.inputs[0], node.inputs[1])
        #scale_gradient = batch_normalization_gradient_of_scale_op(output_grad, node.inputs[0], node.inputs[1])
        #bias_gradient = batch_normalization_gradient_of_bias_op(output_grad, node.inputs[0], node.inputs[1])
        
        return [data_gradient, scale_gradient, bias_gradient]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]

class Batch_Normalization_GradientOp(Op):
    def __call__(self, out_gradient, in_node, bn_scale):
        new_node = Op.__call__(self)
        new_node.inputs = [out_gradient, in_node, bn_scale]
        new_node.tmp_gradient_in_arr = []
        new_node.tmp_gradient_bn_bias = []
        new_node.tmp_gradient_bn_scale = []

        if NAME_RULE==0:
          new_node.name = "(%s,%s,%s)" % (out_gradient.name, in_node.name, bn_scale.name)
        elif NAME_RULE==1:
          new_node.name = "Batch_Normalization_Gradient_of_DataOP"
        else:
          new_node.name = "Batch_Normalization_Gradient_of_DataOP"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s, %s)" % (out_gradient.name, in_node.name, bn_scale.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        
        if use_numpy:
            raise NotImplementedError
        else:
            shapebn = input_vals[2].shape
            node.tmp_gradient_bn_scale = ndarray.empty(shape = shapebn, ctx = input_vals[0].ctx)
            node.tmp_gradient_bn_bias = ndarray.empty(shape = shapebn, ctx = input_vals[0].ctx)
            node.tmp_gradient_in_arr = ndarray.empty(shape = input_vals[1].shape, ctx = input_vals[0].ctx)
            gpu_op.CuDNN_Batch_Normalization_gradient(input_vals[0], input_vals[1], input_vals[2], node.tmp_gradient_in_arr, node.tmp_gradient_bn_scale, node.tmp_gradient_bn_bias)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return (1,)   

class Batch_Normalization_Gradient_of_DataOp(Op):
    def __call__(self, bn_gradient, in_arr):
        new_node = Op.__call__(self)
        new_node.inputs = [bn_gradient, in_arr]

        if NAME_RULE==0:
          new_node.name = "(%s, %s)" % (bn_gradient.name, in_arr.name)
        elif NAME_RULE==1:
          new_node.name = "Batch_Normalization_Gradient_of_DataOP"
        else:
          new_node.name = "Batch_Normalization_Gradient_of_DataOP"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (bn_gradient.name, in_arr.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        
        if use_numpy:
            raise NotImplementedError
        else:
            node.inputs[0].tmp_gradient_in_arr.copyto(output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]        

class Batch_Normalization_Gradient_of_ScaleOp(Op):
    def __call__(self, bn_gradient, in_scale):
        new_node = Op.__call__(self)
        new_node.inputs = [bn_gradient, in_scale]

        if NAME_RULE==0:
          new_node.name = "(%s, %s)" % (bn_gradient.name, in_scale.name)
        elif NAME_RULE==1:
          new_node.name = "Batch_Normalization_Gradient_of_ScaleOP"
        else:
          new_node.name = "Batch_Normalization_Gradient_of_ScaleOP"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (bn_gradient.name, in_scale.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        
        if use_numpy:
            raise NotImplementedError
        else:
            node.inputs[0].tmp_gradient_bn_scale.copyto(output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]        

class Batch_Normalization_Gradient_of_BiasOp(Op):
    def __call__(self, bn_gradient, in_bias):
        new_node = Op.__call__(self)
        new_node.inputs = [bn_gradient, in_bias]

        if NAME_RULE==0:
          new_node.name = "(%s, %s)" % (bn_gradient.name, in_bias.name)
        elif NAME_RULE==1:
          new_node.name = "Batch_Normalization_Gradient_of_BiasOP"
        else:
          new_node.name = "Batch_Normalization_Gradient_of_BiasOP"+str(new_node.id)
          new_node.desc = new_node.name+"(%s, %s)" % (bn_gradient.name, in_bias.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        
        if use_numpy:
            raise NotImplementedError
        else:
            node.inputs[0].tmp_gradient_bn_bias.copyto(output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]         

class PadOp(Op):
    def __call__(self, node_A, paddings, mode = "CONSTANT", constant_values = 0):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.paddings = paddings
        self.mode = mode
        self.constant_values = constant_values
        if NAME_RULE==0:
            new_node.name = "Pad(%s)" % (node_A.name)
        elif NAME_RULE==1:
            new_node.name = "Pad"
        else:
            new_node.name = "pad" + str(new_node.id)
            new_node.desc = new_node.name + "(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            raise NotImplementedError
        else:
          gpu_op.pad(input_vals[0], output_val, self.paddings, self.mode, self.constant_values)

    def gradient(self, node, output_grad):
        return [pad_gradient_op(output_grad, self.paddings, self.mode)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        out_shape = list(input_shapes[0])
        pad_len = len(self.paddings)
        for i in range(4):
            if(i - (4 - pad_len) >= 0):
                out_shape[i] = out_shape[i] + self.paddings[i - (4 - pad_len)][0] + self.paddings[i - (4 - pad_len)][1]
        return tuple(out_shape)

class PadOp_call():
    def __call__(self, node_A, paddings, mode = "CONSTANT", constant_values = 0):
        new_op = PadOp()
        return new_op(node_A, paddings, mode = "CONSTANT", constant_values = 0)

class Pad_GradientOp(Op):
    def __call__(self, node_A, paddings, mode = "CONSTANT"):
        """Creates a node that represents np.sum(node_A, axis=0).
        Only support common-case axis=0 reduction for simplicity of gradient.
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.paddings = paddings
        self.mode = mode
        if NAME_RULE==0:
            new_node.name = "Pad_Gradient(%s)" % (node_A.name)
        elif NAME_RULE==1:
            new_node.name = "Pad_Gradient"
        else:
            new_node.name = "Pad_Gradient" + str(new_node.id)
            new_node.desc = new_node.name + "(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 1
        if use_numpy:
            raise NotImplementedError
        else:
            gpu_op.pad_gradient(input_vals[0], output_val, self.paddings, self.mode)

    def gradient(self, node, output_grad):
        raise NotImplementedError 

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 1
        out_shape = list(input_shapes[0])
        pad_len = len(self.paddings)
        for i in range(4):
            if(i - (4 - pad_len) >= 0):
                out_shape[i] = out_shape[i] - self.paddings[i - (4 - pad_len)][0] - self.paddings[i - (4 - pad_len)][1]
        return tuple(out_shape)

class Pad_GradientOp_call():
    def __call__(self, node_A, paddings, mode = "CONSTANT"):
        new_op = Pad_GradientOp()
        return new_op(node_A, paddings, mode = "CONSTANT")


# Y = ad.concat_op(A, B, axis = 0)


class ConcatOp(Op):
    def __call__(self, node_A, node_B, axis = 0):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        self.axis = axis
        if NAME_RULE==0:
          new_node.name = "Concat(%s, %s)" % (node_A.name, node_B.name)
        elif NAME_RULE==1:
          new_node.name = "Concat"
        else:
          new_node.name = "Concat" + str(new_node.id)
          new_node.desc = new_node.name + "(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            raise NotImplementedError
        else:
          gpu_op.concat(input_vals[0], input_vals[1], output_val, self.axis)

    def gradient(self, node, output_grad):
        return [concat_gradient_op(output_grad, node.inputs[0], self.axis, idx = 0),
                concat_gradient_op(output_grad, node.inputs[1], self.axis, idx = 1)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        out_shape = list(input_shapes[0])
        out_shape[self.axis] = out_shape[self.axis] + input_shapes[1][self.axis]

        return tuple(out_shape) 

class ConcatOP_call():
    def __call__(self, node_A, node_B,axis):
        new_op = ConcatOp()
        return new_op(node_A, node_B, axis)

class Concat_gradientOP(Op):
    def __call__(self, grad_node, input_node, axis, idx):
        new_node = Op.__call__(self)
        new_node.inputs = [grad_node, input_node]
        self.axis = axis
        self.idx = idx
        if NAME_RULE == 0:
            new_node.name = "Concat_gradient(%s, %s)" % (grad_node.name, input_node.name)
        elif NAME_RULE == 1:
            new_node.name = "Concat_gradient"
        else:
            new_node.name = "Concat_gradient" + str(new_node.id)
            new_node.desc = new_node.name + "(%s, %s)" % (grad_node.name, input_node.name)
        return new_node
    def compute(self, node, input_vals, output_val, use_numpy=True):
        assert len(input_vals) == 2
        if use_numpy:
            raise NotImplementedError
        else:
            gpu_op.concat_gradient(input_vals[0], output_val, self.axis, self.idx)
    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        return input_shapes[1]


class Concat_gradientOP_call():
    def __call__(self, node_A, node_B, axis, idx):
        new_op = Concat_gradientOP()
        return new_op(node_A, node_B, axis, idx)

# class Batch_Normalization_Gradient_of_DataOp(Op):
#     def __call__(self, out_gradient, in_node, bn_scale):
#         new_node = Op.__call__(self)
#         new_node.inputs = [out_gradient, in_node, bn_scale]

#         if NAME_RULE==0:
#           new_node.name = "(%s,%s,%s)" % (out_gradient.name, in_node.name, bn_scale.name)
#         elif NAME_RULE==1:
#           new_node.name = "Batch_Normalization_Gradient_of_DataOP"
#         else:
#           new_node.name = "Batch_Normalization_Gradient_of_DataOP"+str(new_node.id)
#           new_node.desc = new_node.name+"(%s, %s, %s)" % (out_gradient.name, in_node.name, bn_scale.name)
#         return new_node

#     def compute(self, node, input_vals, output_val, use_numpy=True):
        
#         if use_numpy:
#             raise NotImplementedError
#         else:
#             shapebn = input_vals[2].shape
#             tmp_gradient_bn_scale = ndarray.empty(shape = shapebn, ctx = ndarray.gpu(0))
#             tmp_gradient_bn_bias = ndarray.empty(shape = shapebn, ctx = ndarray.gpu(0))
#             gpu_op.CuDNN_Batch_Normalization_gradient(input_vals[0], input_vals[1], input_vals[2], output_val, tmp_gradient_bn_scale, tmp_gradient_bn_bias)
#             del tmp_gradient_bn_scale
#             del tmp_gradient_bn_bias

#     def gradient(self, node, output_grad):
#         raise NotImplementedError

#     def infer_shape(self, node, input_shapes):
#         return input_shapes[0]        

# class Batch_Normalization_Gradient_of_ScaleOp(Op):
#     def __call__(self, out_gradient, in_node, bn_scale):
#         new_node = Op.__call__(self)
#         new_node.inputs = [out_gradient, in_node, bn_scale]

#         if NAME_RULE==0:
#           new_node.name = "(%s,%s,%s)" % (out_gradient.name, in_node.name, bn_scale.name)
#         elif NAME_RULE==1:
#           new_node.name = "Batch_Normalization_Gradient_of_ScaleOP"
#         else:
#           new_node.name = "Batch_Normalization_Gradient_of_ScaleOP"+str(new_node.id)
#           new_node.desc = new_node.name+"(%s, %s, %s)" % (out_gradient.name, in_node.name, bn_scale.name)
#         return new_node

#     def compute(self, node, input_vals, output_val, use_numpy=True):
        
#         if use_numpy:
#             raise NotImplementedError
#         else:
#             tmp_gradient_bn_bias = ndarray.empty(shape = input_vals[2].shape, ctx = ndarray.gpu(0))
#             tmp_gradient_in_arr = ndarray.empty(shape = input_vals[1].shape, ctx = ndarray.gpu(0))
#             gpu_op.CuDNN_Batch_Normalization_gradient(input_vals[0], input_vals[1], input_vals[2], tmp_gradient_in_arr, output_val, tmp_gradient_bn_bias)
#             del tmp_gradient_bn_bias
#             del tmp_gradient_in_arr

#     def gradient(self, node, output_grad):
#         raise NotImplementedError

#     def infer_shape(self, node, input_shapes):
#         return input_shapes[2]        

# class Batch_Normalization_Gradient_of_BiasOp(Op):
#     def __call__(self, out_gradient, in_node, bn_scale):
#         new_node = Op.__call__(self)
#         new_node.inputs = [out_gradient, in_node, bn_scale]

#         if NAME_RULE==0:
#           new_node.name = "(%s,%s,%s)" % (out_gradient.name, in_node.name, bn_scale.name)
#         elif NAME_RULE==1:
#           new_node.name = "Batch_Normalization_Gradient_of_BiasOP"
#         else:
#           new_node.name = "Batch_Normalization_Gradient_of_BiasOP"+str(new_node.id)
#           new_node.desc = new_node.name+"(%s, %s, %s)" % (out_gradient.name, in_node.name, bn_scale.name)
#         return new_node

#     def compute(self, node, input_vals, output_val, use_numpy=True):
        
#         if use_numpy:
#             raise NotImplementedError
#         else:
#             tmp_gradient_bn_scale = ndarray.empty(shape = input_vals[2].shape, ctx = ndarray.gpu(0))
#             tmp_gradient_in_arr = ndarray.empty(shape = input_vals[1].shape, ctx = ndarray.gpu(0))
#             gpu_op.CuDNN_Batch_Normalization_gradient(input_vals[0], input_vals[1], input_vals[2], tmp_gradient_in_arr, tmp_gradient_bn_scale, output_val)
#             del tmp_gradient_bn_scale
#             del tmp_gradient_in_arr

#     def gradient(self, node, output_grad):
#         raise NotImplementedError

#     def infer_shape(self, node, input_shapes):
#         return input_shapes[2]        


class Distributed_CommunicateOp(Op):
    def __call__(self,nodeA):
        new_node = Op.__call__(self)
        new_node.inputs = [nodeA]
        new_node.name = "Distributed_Communicate(%s)" % (nodeA.name)
        # print nodeA.name
        return new_node
    
    def compute(self,node,input_vals, output_val, use_numpy = True):
        after_reduce_gradient_cpu = ndarray.empty(shape = output_val.shape, ctx = ndarray.cpu(0))
        if use_numpy:
            gradient_val_cpu = ndarray.array(input_vals[0], ctx = ndarray.cpu(0))
        else:
            gradient_val_cpu = ndarray.array(input_vals[0].asnumpy(), ctx = ndarray.cpu(0))
        # print gradient_val_cpu.asnumpy()
        lib_communicate.DL_Communicate_Init(gradient_val_cpu.handle)
        lib_communicate.DL_Communicate(gradient_val_cpu.handle, after_reduce_gradient_cpu.handle)
        # print after_reduce_gradient_cpu.asnumpy()
        if use_numpy:
            output_val[:] = after_reduce_gradient_cpu.asnumpy()
        else:
            after_reduce_gradient_cpu.copyto(output_val)
    
    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]
# Create global singletons of operators.
add_op = AddOp()
mul_op = MulOp()
add_byconst_op = AddByConstOp()
mul_byconst_op = MulByConstOp()
matmul_op = MatMulOp()
placeholder_op = PlaceholderOp()
oneslike_op = OnesLikeOp()
zeroslike_op = ZerosLikeOp()
reducesumaxiszero_op = ReduceSumAxisZeroOp()
broadcastto_op = BroadcastToOp()
softmaxcrossentropy_op = SoftmaxCrossEntropyOp()
softmax_op = SoftmaxOp()
relu_op = ReluOp()
relu_gradient_op = ReluGradientOp()
conv2d_op = Conv2dOp_call()
conv2d_gradient_of_data_op = Conv2d_Gradient_of_DataOp_call()
conv2d_gradient_of_filter_op = Conv2d_Gradient_of_FilterOp_call()
avg_pool2d_op = Avg_Pool2dOp()
avg_pool2d_gradient_op = Avg_Pool2d_GradientOp()
array_reshape_op = Array_ReshapeOp()
array_reshape_gradient_op = Array_Reshape_GradientOp()
# max_pool2d_op = Max_Pool2dOp()
# max_pool2d_gradient_op = Max_Pool2d_GradientOp()
max_pool2d_op = Max_Pool2dOp()
max_pool2d_gradient_op = Max_Pool2d_GradientOp()
conv2d_reducesum_op = Conv2d_ReduceSumOp()
conv2d_broadcastto_op = Conv2d_BroadcastToOp()
distributed_communicate_op = Distributed_CommunicateOp()
batch_normalization_op = Batch_NormalizationOp()
batch_normalization_gradient_op = Batch_Normalization_GradientOp()
batch_normalization_gradient_of_data_op = Batch_Normalization_Gradient_of_DataOp()
batch_normalization_gradient_of_scale_op = Batch_Normalization_Gradient_of_ScaleOp()
batch_normalization_gradient_of_bias_op = Batch_Normalization_Gradient_of_BiasOp()
dropout_op = DropoutOp()
dropout_gradient_op = Dropout_GradientOp()
pad_op = PadOp_call()
pad_gradient_op = Pad_GradientOp_call()

concat_op = ConcatOP_call()
concat_gradient_op = Concat_gradientOP_call()

class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""
    def __init__(self, eval_node_list, ctx=None, policy=None):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        ctx: runtime DLContext, default is None which means np.ndarray on cpu
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to ndarray.NDArray allocated for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.eval_node_list = eval_node_list
        self.ctx = ctx
        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.feed_shapes = None
        self.policy = policy
        if self.policy == 'swap':
          self.swap_queue = []

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""
        self.node_to_shape_map = {}
        
        for node in self.topo_order:
          if node in feed_shapes:
            self.node_to_shape_map[node] = feed_shapes[node]
          else:
            # print(node.name)
            input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
            self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes)
        

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.

        Implementation note:
        Option 1: Alloc a ndarray.NDArray per node that persists across run()
        Option 2: Implement a memory pool to reuse memory for nodes of same
                shapes. More details see Lecture 7.

        For both options, self.node_to_arr_map stores node->NDArray mapping to
        allow mapping to persist across multiple executor.run().

        Hint: use ndarray.empty(shape, ctx=self.ctx) to allocate NDArray.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""
        assert (self.ctx is not None)
        #self.infer_shape(feed_shapes)
        self.node_to_arr_map = {}
        for node, shape in self.node_to_shape_map.items():
          if self.policy == 'swap':
            if not node.swap:
              self.node_to_arr_map[node] = ndarray.empty(shape, ctx=self.ctx)
          elif self.policy == 'vdnn':
            self.node_to_arr_map[node] = np.empty(shape)
          else:
            self.node_to_arr_map[node] = ndarray.empty(shape, ctx=self.ctx)
    
    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0
    
        # Assume self.ctx is None implies numpy array and numpy ops.

        use_numpy = self.ctx is None
        node_to_val_map = {}
        for node, value in feed_dict.items():
            if use_numpy:
                # all values passed in feed_dict must be np.ndarray
                assert isinstance(value, np.ndarray)
                node_to_val_map[node] = value
            else:
                # convert values to ndarray.NDArray if necessary
                if isinstance(value, np.ndarray):
                    node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
                elif isinstance(value, ndarray.NDArray):
                    node_to_val_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"
        # print"xxxx"
        # collect shapes for all placeholders
        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        # print feed_shapes
        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        if(not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            if (not use_numpy):
                self.memory_plan(self.feed_shapes)

        for node in self.topo_order:
            if node in node_to_val_map:
                continue
            input_vals = [node_to_val_map[n] for n in node.inputs]
            if use_numpy:
                node_val = np.empty(shape = self.node_to_shape_map[node])
            else:
                node_val = self.node_to_arr_map[node]
            node.op.compute(node, input_vals, node_val, use_numpy)
            node_to_val_map[node] = node_val

        if not use_numpy and convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]
        return [node_to_val_map[n] for n in self.eval_node_list]

    # def run(self, feed_dict, convert_to_numpy_ret_vals=False):
    #     """
    #     Parameters
    #     ----------
    #     feed_dict: a dictionary of node->np.ndarray supplied by user.
    #     convert_to_numpy_ret_vals: whether to convert ret vals to np.array

    #     Returns
    #     -------
    #     A list of values for nodes in eval_node_list. NDArray or np.ndarray.
    #     """
    #     def are_feed_shapes_equal(sa, sb):
    #         if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
    #             return False
    #         unmatched_item = set(sa.items()) ^ set(sb.items())
    #         return len(unmatched_item) == 0

    #     # Assume self.ctx is None implies numpy array and numpy ops.

    #     use_numpy = self.ctx is None
    #     node_to_val_map = {}
    #     for node, value in feed_dict.items():
    #       if self.policy == 'vdnn':
    #         assert isinstance(value, np.ndarray)
    #         node_to_val_map[node] = value
    #       else:
    #         if use_numpy:
    #             # all values passed in feed_dict must be np.ndarray
    #             assert isinstance(value, np.ndarray)
    #             node_to_val_map[node] = value
    #         else:
    #             # convert values to ndarray.NDArray if necessary
    #             if isinstance(value, np.ndarray):
    #                 if self.policy == 'swap':
    #                   if node.swap == True:
    #                     node_to_val_map[node] = value
    #                   else:
    #                     node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
    #                 else:
    #                   node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
    #             elif isinstance(value, ndarray.NDArray):
    #                 node_to_val_map[node] = value
    #             else:
    #                 assert False, "feed_dict value type not supported"

    #     # collect shapes for all placeholders
    #     feed_shapes = {}
    #     for node in node_to_val_map:
    #         feed_shapes[node] = node_to_val_map[node].shape

    #     # infer shape if feed_shapes changed since last run
    #     # e.g. call run() on test data after trainng
    #     # print feed_shapes
    #     if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
    #         self.infer_shape(feed_shapes)
    #         self.feed_shapes = feed_shapes
    #         if not self.policy == 'vdnn':
    #           # plan memory if using GPU
    #           if (not use_numpy):
    #             self.memory_plan(feed_shapes)
    #     # Traverse graph in topo order and compute values for all nodes.
    #     global FLAG_SHOW_GRAPH
    #     if self.policy == 'swap':
    #       # generate swap queue
    #       if not use_numpy:
    #         for node in self.topo_order:
    #           if node not in node_to_val_map:
    #             # variable in placeholder
    #             for input_node in node.inputs:
    #               if input_node.swap == True:
    #                 self.swap_queue.append(input_node)
    #             # variable grad
    #             if node.swap == True:
    #                 self.swap_queue.append(node)
    #         node_in_GPU = None
    #         if FLAG_SHOW_GRAPH:
    #           print "Show swap queue:"
    #           for node in self.swap_queue:
    #             print node
    #     elif self.policy == 'vdnn':
    #       # TODO traverse graph to select in-gpu window
    #       window = [0,0]
    #       if not use_numpy:
    #         nvmlInit()
    #         handle = nvmlDeviceGetHandleByIndex(0)
    #         info = nvmlDeviceGetMemoryInfo(handle)
    #         gpu_mem = info.free
    #         nvmlShutdown()
    #         loss_node = self.eval_node_list[0]
    #         window[1] = self.topo_order.index(loss_node)+1
    #         window[0] = self.topo_order.index(loss_node)+1
    #         for node in reversed(self.topo_order[:window[1]+1]):
    #           node_size = 4  # float32
    #           #print node, self.node_to_shape_map[node]
    #           for shape in self.node_to_shape_map[node]:
    #             node_size = node_size * shape
    #           if gpu_mem > node_size:
    #             gpu_mem = gpu_mem - node_size
    #             window[0] = window[0] - 1
    #         #print "gpu_mem:",gpu_mem
    #     # Traverse graph in topo order and compute values for all nodes. 
    #     if FLAG_SHOW_GRAPH:
    #       print "run topo_order"
    #     # Show graph dependency
    #     if FLAG_SHOW_GRAPH:
    #       print "node:",node
    #       print "node.desc:",node.desc

    #     for node in self.topo_order:
    #       if self.policy == 'vdnn':
    #         # Skip placeholder nodes
    #         if node in node_to_val_map:
    #           continue
    #         # H2D before compute
    #         ## Collect inputs 
    #         input_vals = []
    #         for n in node.inputs:
    #           if not use_numpy: 
    #             if isinstance(node_to_val_map[n], np.ndarray):
    #               node_to_val_map[n] = ndarray.array(node_to_val_map[n], ctx=self.ctx)
    #           input_vals.append(node_to_val_map[n])
    #         ## Alloc node space
    #         if use_numpy:
    #             node_val = np.empty(shape=self.node_to_shape_map[node])
    #         else:
    #             node_val = ndarray.empty(shape=self.node_to_shape_map[node], ctx=self.ctx)
    #         # Compute
    #         # node_val is modified in-place whether np.ndarray or NDArray
    #         node.op.compute(node, input_vals, node_val, use_numpy)
    #         # D2H after compute
    #         if use_numpy:
    #           node_to_val_map[node] = node_val
    #         else:
    #           node_index = self.topo_order.index(node)
    #           if node_index > window[0] and node_index < window[1]:
    #             node_to_val_map[node] = node_val
    #             continue
    #           node_to_val_map[node] = node_val.asnumpy()
    #           del node_val
    #           for n in node.inputs:
    #             if isinstance(node_to_val_map[n], ndarray.NDArray):
    #               tmp_val = node_to_val_map[n].asnumpy()
    #               del node_to_val_map[n]
    #               node_to_val_map[n] = tmp_val
    #       elif self.policy == 'swap':
    #         # Switch in GPU
    #         if not use_numpy:
    #           if self.swap_queue and (node_in_GPU==None):
    #             swap_node = self.swap_queue[0]
    #             if swap_node in node_to_val_map:
    #               node_to_val_map[swap_node] = ndarray.array(node_to_val_map[swap_node], ctx=self.ctx)
    #             else:
    #               self.node_to_arr_map[swap_node] = ndarray.empty(self.node_to_shape_map[swap_node], ctx=self.ctx) 
    #             node_in_GPU = swap_node.id
            
    #         if node in node_to_val_map:
    #             # Skip placeholder nodes. Values already provided by feed_dict.
    #             continue
    #         # Compute
    #         input_vals = [node_to_val_map[n] for n in node.inputs]
    #         if use_numpy:
    #             node_val = np.empty(shape=self.node_to_shape_map[node])
    #         else:
    #             node_val = self.node_to_arr_map[node]
    #         # node_val is modified in-place whether np.ndarray or NDArray
    #         node.op.compute(node, input_vals, node_val, use_numpy)
    #         if node.swap == True:
    #           node_to_val_map[node] = node_val.asnumpy()
    #           del node_val
    #           del self.node_to_arr_map[node]
    #           del self.swap_queue[0]
    #           node_in_GPU = None
    #         else:
    #           node_to_val_map[node] = node_val
    #           # Switch out GPU
    #           if not use_numpy:
    #             if self.swap_queue:
    #               if self.swap_queue[0] in node.inputs:
    #                 out_node = self.swap_queue.pop(0)
    #                 if self.swap_queue:
    #                   if not self.swap_queue[0].id == node_in_GPU:
    #                     tmp_array = node_to_val_map[out_node].asnumpy()
    #                     del node_to_val_map[out_node]
    #                     node_to_val_map[out_node] = tmp_array
    #                     node_in_GPU = None
    #       else:
    #         if node in node_to_val_map:
    #             # Skip placeholder nodes. Values already provided by feed_dict.
    #             continue
    #         input_vals = [node_to_val_map[n] for n in node.inputs]
    #         # print self.node_to_shape_map[node]

    #         if use_numpy:
    #             node_val = np.empty(shape=self.node_to_shape_map[node])
    #         else:
    #             node_val = self.node_to_arr_map[node]
    #         # node_val is modified in-place whether np.ndarray or NDArray
    #         # if (len(node.inputs) == 1):
    #         #     print "computs",node.inputs[0].name
    #         # else:
    #         #     print "computs",node.inputs[0].name,node.inputs[1].name
    #         # print node.name
                                
    #         # print node_val.shape
    #         # print "xxx"
    #         # print node.name
    #         node.op.compute(node, input_vals, node_val, use_numpy)
    #         # print "xxx"
    #         node_to_val_map[node] = node_val
    #         # print "xxx"

    #     if FLAG_SHOW_GRAPH:
    #       FLAG_SHOW_GRAPH = False
    #     # Collect node values.
    #     if not use_numpy and convert_to_numpy_ret_vals:
    #         if self.policy == 'swap':
    #           node_values = []
    #           for n in self.eval_node_list:
    #             if n.swap == True:
    #               node_values.append(node_to_val_map[n])
    #             else:
    #               node_values.append(node_to_val_map[n].asnumpy())
    #           return node_values
    #         elif self.policy == 'vdnn':
    #           return [node_to_val_map[n] for n in self.eval_node_list]
    #         else:             
    #           return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]  
    #     return [node_to_val_map[n] for n in self.eval_node_list]

def gradients(output_node, node_list, scheduler_policy=None):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        #print len(node.name)
        #print len(node.inputs)
        #raw_input("\n\nPress the enter key to exit.")
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            # print node.name
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])
    if scheduler_policy == 'swap':
      for node in node_list:
        if node.swap:
          node_to_output_grad[node].swap=True
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    # grad_node_list = [distributed_communicate_op(node_to_output_grad[node]) for node in node_list]
    return grad_node_list

def distributed_gradients(output_node, node_list, scheduler_policy=None):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        #print len(node.name)
        #print len(node.inputs)
        #raw_input("\n\nPress the enter key to exit.")
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])
    if scheduler_policy == 'swap':
      for node in node_list:
        if node.swap:
          node_to_output_grad[node].swap=True
    # grad_node_list = [node_to_output_grad[node] for node in node_list]
    grad_node_list = [distributed_communicate_op(node_to_output_grad[node]) for node in node_list]
    return grad_node_list

##################
# Helper Methods #
##################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
    """Return output shape of broadcast shape_a, shape_b.
    e.g. broadcast_rule((3,2), (4,3,2))
    returns output_shape = (4,3,2)

    Check out explanations and more examples at
    https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    """
    assert(isinstance(shape_a, tuple))
    assert(isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)
    
