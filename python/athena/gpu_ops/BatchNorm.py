from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory
import numpy as np

class Batch_NormalizationOp(Op):
    def __call__(self, node_in, bn_scale, bn_bias, momentum=0.99, eps=0.01):
        new_node = Op.__call__(self)
        new_node.inputs = [node_in, bn_scale, bn_bias]
        self.momentum = momentum
        self.eps = eps
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s,%s,%s)" % (
                node_in.name, bn_scale.name, bn_bias.name)
        elif NAME_RULE == 1:
            new_node.name = "Batch_NormalizationOp"
        else:
            new_node.name = "Batch_NormalizationOp" + str(new_node.id)
            new_node.desc = new_node.name + \
                            "(%s, %s, %s)" % (node_in.name, bn_scale.name, bn_bias.name)
        return new_node

    def profile(self, node, input_vals, output_val, is_static = True):

        assert len(input_vals) == 3

        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape) + \
                                         get_array_memory(input_vals[1].shape) + \
                                         get_array_memory(input_vals[2].shape)
            # output memory
            node.profiler.output_memory = get_array_memory(output_val.shape)
            # no workspace
            node.profiler.workspace_memory = 0
            # execute time
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            # import time
            # start = time.time()
            from athena import ndarray
            from ..gpu_links import CuDNN_Batch_Normalization

            CuDNN_Batch_Normalization(
                input_vals[0], input_vals[1], input_vals[2], output_val,
                None, None, self.momentum,
                self.eps, None, node.profiler)
            # node.profiler.time = time.time() - start

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        if use_numpy:
            from .._base import DNNL_LIB
            # print(output_val.shape)
            if DNNL_LIB['DnnlBatchNorm']:
                from ..cpu_links import batch_norm as cpu_batch_norm
                from ..ndarray import numpyasdlarrayhandle
                if node.save_mean is None:
                    C = input_vals[0].shape[1]
                    node.save_mean = np.zeros([C], dtype=np.float32)
                    node.save_var = np.zeros([C], dtype=np.float32)
                input = numpyasdlarrayhandle(input_vals[0])
                bn_scale = numpyasdlarrayhandle(input_vals[1])
                bn_bias = numpyasdlarrayhandle(input_vals[2])
                output = numpyasdlarrayhandle(output_val)
                mean = numpyasdlarrayhandle(node.save_mean)
                var = numpyasdlarrayhandle(node.save_var)
                cpu_batch_norm(input, bn_scale, bn_bias, output, mean, var, self.momentum, self.eps)
            else:
                output_val[:], node.save_mean, node.save_var = batchnorm_forward(input_vals[0], input_vals[1],
                                                                                 input_vals[2], node.save_mean,
                                                                                 node.save_var, self.momentum, self.eps)
        else:
            from ..gpu_links import CuDNN_Batch_Normalization
            from athena import ndarray
            
            CuDNN_Batch_Normalization(
                input_vals[0], input_vals[1], input_vals[2], output_val,
                None, None, self.momentum,
                self.eps, stream_handle, None)


    def gradient(self, node, output_grad):

        data_gradient = batch_normalization_gradient_op(
            output_grad, node.inputs[0], node.inputs[1], self.eps)
        
        # from .ZerosLike import zeroslike_op

        return [data_gradient, None, None]

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class Batch_Normalization_GradientOp(Op):
    def __call__(self, out_gradient, in_node, bn_scale, eps):
        new_node = Op.__call__(self)
        new_node.inputs = [out_gradient, in_node, bn_scale]
        self.tmp_gradient_in_arr = None
        self.tmp_gradient_bn_bias = None
        self.tmp_gradient_bn_scale = None
        self.eps = eps
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s,%s,%s)" % (
                out_gradient.name, in_node.name, bn_scale.name)
        elif NAME_RULE == 1:
            new_node.name = "Batch_Normalization_Gradient_OP"
        else:
            new_node.name = "Batch_Normalization_Gradient_OP" + \
                            str(new_node.id)
            new_node.desc = new_node.name + \
                            "(%s, %s, %s)" % (out_gradient.name, in_node.name, bn_scale.name)
        return new_node

    def update_mean_and_var(self, saved_mean, saved_var):
        self.saved_mean = saved_mean
        self.saved_var = saved_var

    def profile(self, node, input_vals, output_val, is_static = True):

        assert len(input_vals) == 3
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape) + \
                                         get_array_memory(input_vals[1].shape) + \
                                         get_array_memory(input_vals[2].shape)
            # output memory
            node.profiler.output_memory = get_array_memory(output_val.shape)
            # no workspace
            node.profiler.workspace_memory = get_array_memory(input_vals[2].shape) * 2 + \
                                             get_array_memory(input_vals[1].shape)
            # execute time
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            import time
            start = time.time()
            from .. import ndarray
            from ..gpu_links import CuDNN_Batch_Normalization_gradient
            shapebn = input_vals[2].shape

            if self.tmp_gradient_bn_bias == None:
                self.tmp_gradient_bn_bias = ndarray.empty(
                    shape=shapebn, ctx=input_vals[0].ctx)
            else:
                self.tmp_gradient_bn_bias.malloc_itself()
                    
            if self.tmp_gradient_bn_scale == None:
                self.tmp_gradient_bn_scale = ndarray.empty(
                    shape=shapebn, ctx=input_vals[0].ctx)
            else:
                self.tmp_gradient_bn_scale.malloc_itself()
                    
            CuDNN_Batch_Normalization_gradient(input_vals[0], input_vals[1], input_vals[2],
                                               output_val, self.tmp_gradient_bn_scale,
                                               self.tmp_gradient_bn_bias, None,
                                               None, self.eps, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 3
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlBatchNorm_Gradient']:
                from ..cpu_links import batch_norm_gradient as cpu_batch_norm_gradient
                from ..ndarray import numpyasdlarrayhandle
                import numpy as np
                if node.tmp_gradient_bn_bias is None:
                    shapebn = input_vals[2].shape
                    node.tmp_gradient_bn_bias = np.zeros(shape=shapebn, dtype=np.float32)
                    node.tmp_gradient_bn_scale = np.zeros(shape=shapebn, dtype=np.float32)
                    node.tmp_gradient_in_arr = np.zeros(shape=input_vals[1].shape, dtype=np.float32)

                # print("Attention: bn_grad_shape", input_vals[0].shape, input_vals[1].shape, input_vals[2].shape)
                grad_y = numpyasdlarrayhandle(input_vals[0])
                input_x = numpyasdlarrayhandle(input_vals[1])
                bn_scale = numpyasdlarrayhandle(input_vals[2])
                # print(input_vals[1].shape)
                # print(input_vals[2].shape)
                bias = np.zeros(shape=input_vals[2].shape)
                bn_bias = numpyasdlarrayhandle(bias)
                # output = numpyasdlarrayhandle(output_val)
                tmp_bias = numpyasdlarrayhandle(node.tmp_gradient_bn_bias)
                tmp_scale = numpyasdlarrayhandle(node.tmp_gradient_bn_scale)
                tmp_arr = numpyasdlarrayhandle(node.tmp_gradient_in_arr)
                mean = numpyasdlarrayhandle(self.forward_node.save_mean)
                var = numpyasdlarrayhandle(self.forward_node.save_var)

                cpu_batch_norm_gradient(grad_y, input_x, bn_scale, bn_bias, tmp_arr, tmp_scale,
                                        tmp_bias, mean, var, self.eps)
            else:
                import numpy as np
                if node.tmp_gradient_bn_bias is None:
                    typebn = input_vals[2].dtype
                    shapebn = input_vals[2].shape
                    node.tmp_gradient_bn_bias = np.zeros(shape=shapebn, dtype=typebn)
                    node.tmp_gradient_bn_scale = np.zeros(shape=shapebn, dtype=typebn)
                node.tmp_gradient_in_arr, node.tmp_gradient_bn_scale, node.tmp_gradient_bn_bias = batchnorm_backward(
                    input_vals[0], input_vals[1], input_vals[2], node.tmp_gradient_bn_scale, node.tmp_gradient_bn_bias,
                    self.eps, self.forward_node.save_mean, self.forward_node.save_var)
        else:
            from .. import ndarray
            from ..gpu_links import CuDNN_Batch_Normalization_gradient

            shapebn = input_vals[2].shape
            if self.tmp_gradient_bn_bias == None:
                self.tmp_gradient_bn_bias = ndarray.empty(
                    shape=shapebn, ctx=input_vals[0].ctx)
            else:
                self.tmp_gradient_bn_bias.malloc_itself()
    
            if self.tmp_gradient_bn_scale == None:
                self.tmp_gradient_bn_scale = ndarray.empty(
                    shape=shapebn, ctx=input_vals[0].ctx)
            else:
                self.tmp_gradient_bn_scale.malloc_itself()

            CuDNN_Batch_Normalization_gradient(input_vals[0], input_vals[1], input_vals[2],
                                               output_val, self.tmp_gradient_bn_scale,
                                               self.tmp_gradient_bn_bias, None,
                                               None, self.eps, stream_handle, None)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


class Batch_Normalization_Gradient_of_DataOp(Op):
    def __call__(self, bn_gradient, in_arr):
        new_node = Op.__call__(self)
        new_node.inputs = [bn_gradient, in_arr]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()

        if NAME_RULE == 0:
            new_node.name = "(%s, %s)" % (bn_gradient.name, in_arr.name)
        elif NAME_RULE == 1:
            new_node.name = "Batch_Normalization_Gradient_of_DataOP"
        else:
            new_node.name = "Batch_Normalization_Gradient_of_DataOP" + \
                            str(new_node.id)
            new_node.desc = new_node.name + \
                            "(%s, %s)" % (bn_gradient.name, in_arr.name)
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
            # # # #
            # TODO copy bandwidth
            # # # #
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            import time
            start = time.time()
            node.inputs[0].tmp_gradient_in_arr.copyto(output_val)
            node.profiler.time = (time.time() - start) * 1000
            node.inputs[0].tmp_gradient_in_arr.delete_itself()

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        if use_numpy:
            output_val[:] = node.inputs[0].tmp_gradient_in_arr
        else:
            node.inputs[0].tmp_gradient_in_arr.copyto(output_val, stream_handle)
            node.inputs[0].tmp_gradient_in_arr.delete_itself()

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]


class Batch_Normalization_Gradient_of_ScaleOp(Op):
    def __call__(self, bn_gradient, in_scale):
        new_node = Op.__call__(self)
        new_node.inputs = [bn_gradient, in_scale]
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()

        if NAME_RULE == 0:
            new_node.name = "(%s, %s)" % (bn_gradient.name, in_scale.name)
        elif NAME_RULE == 1:
            new_node.name = "Batch_Normalization_Gradient_of_ScaleOP"
        else:
            new_node.name = "Batch_Normalization_Gradient_of_ScaleOP" + \
                            str(new_node.id)
            new_node.desc = new_node.name + \
                            "(%s, %s)" % (bn_gradient.name, in_scale.name)
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
            # # # #
            # TODO copy bandwidth
            # # # #
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            import time
            start = time.time()
            node.inputs[0].tmp_gradient_bn_scale.copyto(output_val)
            node.profiler.time = (time.time() - start) * 1000
            node.inputs[0].tmp_gradient_bn_scale.delete_itself()

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        if use_numpy:
            output_val[:] = node.inputs[0].tmp_gradient_bn_scale
        else:
            node.inputs[0].tmp_gradient_bn_scale.copyto(output_val, stream_handle)
            node.inputs[0].tmp_gradient_bn_scale.delete_itself()

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]


class Batch_Normalization_Gradient_of_BiasOp(Op):
    def __call__(self, bn_gradient, in_bias):
        new_node = Op.__call__(self)
        new_node.inputs = [bn_gradient, in_bias]

        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s, %s)" % (bn_gradient.name, in_bias.name)
        elif NAME_RULE == 1:
            new_node.name = "Batch_Normalization_Gradient_of_BiasOP"
        else:
            new_node.name = "Batch_Normalization_Gradient_of_BiasOP" + \
                            str(new_node.id)
            new_node.desc = new_node.name + \
                            "(%s, %s)" % (bn_gradient.name, in_bias.name)
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
            # # # #
            # TODO copy bandwidth
            # # # #
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            import time
            start = time.time()
            node.inputs[0].tmp_gradient_bn_bias.copyto(output_val)
            node.profiler.time = (time.time() - start) * 1000
            node.inputs[0].tmp_gradient_bn_bias.delete_itself()

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        if use_numpy:
            output_val[:] = node.inputs[0].tmp_gradient_bn_bias
        else:
            node.inputs[0].tmp_gradient_bn_bias.copyto(output_val, stream_handle)
            node.inputs[0].tmp_gradient_bn_bias.delete_itself()

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[1]

def batch_normalization_op(node_in, bn_scale, bn_bias, momentum=0.99, eps=0.01):
    """Batch normalization layer node.

    Parameters:
    ----
    node_in : Node
        Input data.
    bn_scale : float
        scaling parameter
    bn_bias :
        learnable bias parameter
    momentum : float
        Acting on the calculation of mean and variance, the mean and variance values in historical batch are retained.
    eps : float
        Epsilon value for numerical stability.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_NormalizationOp()(node_in, bn_scale, bn_bias, momentum, eps)


def batch_normalization_gradient_op(out_gradient, in_node, bn_scale, eps):
    """Gradient node of batch normalization.

    Parameters:
    ----
    out_gradient :
        The gradient array.
    in_node : Node
        Input node of bn layer.
    bn_scale :
        Scaling parameter.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_Normalization_GradientOp()(out_gradient, in_node, bn_scale, eps)


def batch_normalization_gradient_of_data_op(bn_gradient, in_arr):
    """Gradient node of data of  batch normalization.

    Parameters:
    ----
    bn_gradient :
        The gradient array.
    in_arr : Node
        Input array of bn layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_Normalization_Gradient_of_DataOp()(bn_gradient, in_arr)


def batch_normalization_gradient_of_scale_op(bn_gradient, in_scale):
    """Gradient node of scale parameter of batch normalization.

    Parameters:
    ----
    bn_gradient :
        The gradient array.
    in_scale :
        Scaling parameter of bn layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_Normalization_Gradient_of_ScaleOp()(bn_gradient, in_scale)


def batch_normalization_gradient_of_bias_op(bn_gradient, in_bias):
    """Gradient node of bias parameter of batch normalization.

    Parameters:
    ----
    bn_gradient :
        The gradient array.
    in_bias :
        Bias parameter of bn layer.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Batch_Normalization_Gradient_of_BiasOp()(bn_gradient, in_bias)


def batchnorm_forward(x, bn_scale, bn_bias, save_mean, save_var, momentum=0.99, eps=0.01):
    D = x.shape[1]
    if save_mean is None:
        save_mean = np.zeros(D, dtype=x.dtype)
    if save_var is None:
        save_var = np.ones(D, dtype=x.dtype)

    sample_mean = x.mean(axis=(0, 2, 3), dtype=x.dtype)
    sample_var = x.var(axis=(0, 2, 3), dtype=x.dtype)
    save_mean = momentum * save_mean + (1.0 - momentum) * sample_mean
    save_var = momentum * save_var + (1.0 - momentum) * sample_var

    std = np.sqrt(sample_var.reshape(1, D, 1, 1) + eps, dtype=x.dtype)
    x_centered = x - sample_mean.reshape(1, D, 1, 1)
    x_norm = x_centered / std
    # print(bn_bias)
    out = bn_scale.reshape(1, D, 1, 1) * x_norm + bn_bias.reshape(1, D, 1, 1)

    return out, save_mean, save_var


def batchnorm_backward(gradient_Y, x, bn_scale, dbn_scale, dbn_bias, eps, save_mean, save_var):
    D = gradient_Y.shape[1]

    sample_mean = save_mean
    sample_var = save_var

    std = np.sqrt(sample_var.reshape(1, D, 1, 1) + eps)
    x_centered = x - sample_mean.reshape(1, D, 1, 1)
    x_norm = x_centered / std

    dbn_scale = (gradient_Y * x_norm).sum(axis=(0, 2, 3))
    dbn_bias = gradient_Y.sum(axis=(0, 2, 3))

    dx_norm = gradient_Y * bn_scale.reshape(1, D, 1, 1)
    dx_centered = dx_norm / std
    dmean = -(dx_centered.sum(axis=(0, 2, 3)) + 2 / D * x_centered.sum(axis=(0, 2, 3))).reshape(1, D, 1, 1)
    dstd = (dx_norm * x_centered * -std ** (-2)).sum(axis=(0, 2, 3)).reshape(1, D, 1, 1)
    dvar = dstd / 2 / std
    dx = dx_centered + (dmean + dvar * 2 * x_centered) / D

    return dx, dbn_scale, dbn_bias