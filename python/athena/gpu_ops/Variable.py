from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE
from .. import ndarray


def Variable(name, value=None, trainable=True, swap=False):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op(value, trainable)
    placeholder_node.name = name
    if NAME_RULE == 2:
        placeholder_node.desc = name + str(placeholder_node.id)
    placeholder_node.swap = swap
    return placeholder_node


class PlaceholderOp(Op):
    def __call__(self, value=None, trainable=True):
        """Creates a variable node."""
        new_node = Op.__call__(self)
        if value is not None:
            assert isinstance(value, (np.ndarray, ndarray.NDArray))
        else:
            trainable = False
        new_node.tensor_value = value
        new_node.trainable = trainable
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        if node.tensor_value is None:
            assert False, "placeholder %s values provided by feed_dict" % node.name
        else:
            return node.tensor_value

    def gradient(self, node, output_grad):
        return None

    def infer_shape(self, node, input_shapes):
        if node.tensor_value is None:
            assert False, "placeholder %s shape provided by feed_shape" % node.name
        else:
            return node.tensor_value.shape


def placeholder_op(value=None, trainable=True):
    """Node of variable placeholder.

    Parameters:
    ----
    None

    Returns:
    ----
    A new Node instance created by Op.

    """
    return PlaceholderOp()(value, trainable)
