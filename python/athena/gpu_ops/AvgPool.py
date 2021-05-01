from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory

class Avg_Pool2dOp(Op):
    def __call__(self, node_A, kernel_H, kernel_W, padding, stride):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s)" % (node_A.name)
        elif NAME_RULE == 1:
            new_node.name = "Avg_Pool2d"
        else:
            new_node.name = "Avg_Pool2d"+str(new_node.id)
            new_node.desc = new_node.name+"(%s)" % (node_A.name)
        return new_node

    def np_average_pooling(self, input, kernel_H, kernel_W, padding=0, stride=1):
        N, C, H, W = input.shape
        assert((H + 2 * padding - kernel_H) % stride == 0)
        assert((W + 2 * padding - kernel_W) % stride == 0)
        pooled_H = (H + 2 * padding - kernel_H) / stride + 1
        pooled_W = (W + 2 * padding - kernel_W) / stride + 1
        pooled_layer = np.zeros(
            shape=(N, C, pooled_H, pooled_W), dtype=np.float32)
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
            from ..gpu_links import CuDNN_average_pooling2d
            N, C, H, W = input_vals[0].shape
            _N, _C, _H, _W = output_val.shape
            assert(N == _N)
            assert(C == _C)
            assert((H + 2 * self.padding - self.kernel_H) / self.stride + 1 == _H)
            assert((W + 2 * self.padding - self.kernel_W) / self.stride + 1 == _W)
            CuDNN_average_pooling2d(
                input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 1
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlAvgPool']:
                from ..cpu_links import avg_pool as cpu_avg_pool
                from ..ndarray import numpyasdlarrayhandle
                input = numpyasdlarrayhandle(input_vals[0])
                output = numpyasdlarrayhandle(output_val)
                cpu_avg_pool(input, self.kernel_H, self.kernel_W, output, self.padding, self.stride)
            else:
                output_val[:] = self.np_average_pooling(
                    input_vals[0], self.kernel_H, self.kernel_W, self.padding, self.stride)
        else:
            from ..gpu_links import CuDNN_average_pooling2d
            N, C, H, W = input_vals[0].shape
            _N, _C, _H, _W = output_val.shape
            assert(N == _N)
            assert(C == _C)
            assert((H + 2 * self.padding - self.kernel_H) / self.stride + 1 == _H)
            assert((W + 2 * self.padding - self.kernel_W) / self.stride + 1 == _W)
            # average_pooling2d(input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride)
            CuDNN_average_pooling2d(
                input_vals[0], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride, stream_handle, None)

    def gradient(self, node, output_grad):
        return [avg_pool2d_gradient_op(node, output_grad, node.inputs[0], self.kernel_H, self.kernel_W, self.padding, self.stride)]

    def infer_shape(self, node, input_shapes):
        """Need to handle input_vals[0].shape != input_vals[1].shape"""
        """TODO: Your code here"""
        assert len(input_shapes) == 1
        N, C, H, W = input_shapes[0]
        p_H = (H + 2 * self.padding - self.kernel_H) / self.stride + 1
        p_W = (W + 2 * self.padding - self.kernel_W) / self.stride + 1
        return (N, C, p_H, p_W)


class Avg_Pool2d_GradientOp(Op):
    def __call__(self, node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride):
        new_node = Op.__call__(self)
        new_node.inputs = [node_out, node_out_gradient, node_in]
        self.padding = padding
        self.stride = stride
        self.kernel_H = kernel_H
        self.kernel_W = kernel_W
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "(%s, %s, %s)" % (
                node_out.name, node_out_gradient.name, node_in.name)
        elif NAME_RULE == 1:
            new_node.name = "Avg_Pool2d_Gradient"
        else:
            new_node.name = "Avg_Pool2d_Gradient"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s, %s)" % (node_out.name,
                                  node_out_gradient.name, node_in.name)
        return new_node

    def np_average_pooling_gradient(self, gradient_y, kernel_H, kernel_W, padding=0, stride=1):
        N, C, pooled_H, pooled_W = gradient_y.shape
        H = (pooled_H - 1) * stride + kernel_H - 2 * padding
        W = (pooled_W - 1) * stride + kernel_W - 2 * padding

        gradient_x = np.zeros(shape=(N, C, H, W), dtype=np.float32)
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
                                gradient_x[n][c][i][j] += gradient_y[n][c][h][w] / \
                                    pooling_size

        return gradient_x

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
            import time
            start = time.time()
            from ..gpu_links import CuDNN_average_pooling2d_gradient
            N, C, H, W = input_vals[0].shape
            _N, _C, _H, _W = output_val.shape
            assert(N == _N)
            assert(C == _C)
            assert((_H + 2 * self.padding - self.kernel_H) / self.stride + 1 == H)
            assert((_W + 2 * self.padding - self.kernel_W) / self.stride + 1 == W)
            CuDNN_average_pooling2d_gradient(
                input_vals[0], input_vals[1], input_vals[2], self.kernel_H,
                self.kernel_W, output_val, self.padding, self.stride, None, node.profiler)
            node.profiler.time = (time.time() - start) * 1000


    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 3
        if use_numpy:
            # output_val[:] allows modify in-place
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlAvgPool_Gradient']:
                from ..cpu_links import avg_pool_gradient as cpu_avg_pool_gradient
                from ..ndarray import numpyasdlarrayhandle
                input = numpyasdlarrayhandle(input_vals[1])
                output = numpyasdlarrayhandle(output_val)
                cpu_avg_pool_gradient(input, self.kernel_H, self.kernel_W, output, self.padding, self.stride)
            else:
                output_val[:] = self.np_average_pooling_gradient(
                    input_vals[1], self.kernel_H, self.kernel_W, self.padding, self.stride)
        else:
            from ..gpu_links import CuDNN_average_pooling2d_gradient

            N, C, H, W = input_vals[0].shape
            _N, _C, _H, _W = output_val.shape
            assert(N == _N)
            assert(C == _C)
            assert((_H + 2 * self.padding - self.kernel_H) / self.stride + 1 == H)
            assert((_W + 2 * self.padding - self.kernel_W) / self.stride + 1 == W)
            CuDNN_average_pooling2d_gradient(
                input_vals[0], input_vals[1], input_vals[2], self.kernel_H, self.kernel_W, output_val, self.padding, self.stride, stream_handle, None)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 3
        return input_shapes[2]


def avg_pool2d_op(node_A, kernel_H, kernel_W, padding, stride):
    """Average pooling node.

    Parameters:
    ----
    node_A : Node
        Input node.
    kernel_H : 
        Kernel height.
    kernel_W :
        Kernel width.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Avg_Pool2dOp()(node_A, kernel_H, kernel_W, padding, stride)


def avg_pool2d_gradient_op(node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride):
    """Gradient node of average pooling.

    Parameters:
    ----
    node_out : Node
        Output node of average pooling.
    node_out_gradient : Node
        Previous gradient node.
    node_in : Node
        Input node of average pooling.
    kernel_H : 
        Kernel height.
    kernel_W :
        Kernel width.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Avg_Pool2d_GradientOp()(node_out, node_out_gradient, node_in, kernel_H, kernel_W, padding, stride)
