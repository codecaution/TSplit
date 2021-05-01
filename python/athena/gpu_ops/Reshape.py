from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class Array_ReshapeOp(Op):
    def __call__(self, node_A, output_shape):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.output_shape = output_shape
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "Array_Reshape_Op"
        else:
            new_node.name = "Array_Reshape_Op"+str(new_node.id)
            new_node.desc = new_node.name+"(%s)" % (node_A.name)
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
            from ..gpu_links import array_reshape
            array_reshape(input_vals[0], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000
 
    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert(len(input_vals) == 1)
        output_shape = output_val.shape
        self.output_shape = output_shape
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_Reshape']:
                from ..cpu_links import reshape as cpu_reshape
                from ..ndarray import numpyasdlarrayhandle
                input_x = numpyasdlarrayhandle(input_vals[0])
                output = numpyasdlarrayhandle(output_val)
                cpu_reshape(input_x, output)
            else:
                output_val[:] = input_vals[0].reshape(output_shape)
        else:   
            from ..gpu_links import array_reshape
            array_reshape(input_vals[0], output_val, stream_handle, None)


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
            output_size *= output_shape[i]
        if(idx == -1):
            assert input_size == output_size
        else:
            output_size = output_size * (-1)
            assert (input_size % output_size == 0)
            output_shape[idx] = input_size // output_size
        output_shape = tuple(output_shape)
        self.output_shape = output_shape
        return output_shape


class Array_Reshape_GradientOp(Op):
    def __call__(self, node_in, node_out):
        new_node = Op.__call__(self)
        new_node.inputs = [node_in, node_out]

        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s,%s)" % (node_in.name, node_out.name)
        elif NAME_RULE == 1:
            new_node.name = "Array_Reshape_GradientOp"
        else:
            new_node.name = "Array_Reshape_GradientOp"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (node_in.name, node_out.name)
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
            from ..gpu_links import array_reshape
            array_reshape(input_vals[1], output_val, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        # the size of input_array
        shapeIn = input_vals[0].shape
        # print input_vals[1].shape
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['cpu_Reshape']:
                from ..cpu_links import reshape as cpu_reshape
                from ..ndarray import numpyasdlarrayhandle
                input_x = numpyasdlarrayhandle(input_vals[1])
                output = numpyasdlarrayhandle(output_val)
                cpu_reshape(input_x, output)
            else:
                output_val[:] = input_vals[1].reshape(shapeIn)
        else:    
            from ..gpu_links import array_reshape
            array_reshape(input_vals[1], output_val, stream_handle, None)


    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        # print input_shapes[0]
        return input_shapes[0]


def array_reshape_op(node, output_shape):
    """Reshapes an input array without copy.

    Parameters:
    ----
    node : Node
        Input variable.
    output_shape: tuple(int)
        Expected shape of the output array.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Array_ReshapeOp()(node, output_shape)


def array_reshape_gradient_op(node_in, node_out):
    """Gradient of reshape operation.

    Parameters:
    ----
    node_in : Node
        Input node of reshape operation.
    node_out: Node
        Previous gradient node.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Array_Reshape_GradientOp()(node_in, node_out)
