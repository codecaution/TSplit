from __future__ import absolute_import
import numpy as np
from .Node import Op, NAME_RULE, PROFILING_MODE
from .. import profiler
from .._base import get_array_memory


class Conv2dOp(Op):
    # nodeA : x  nodeB : filter
    def __call__(self, node_A, node_B, padding=0, padding2 = None, stride=1, For_ResNet = False):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]

        self.padding = padding
        self.padding2 = padding2
        self.For_ResNet = For_ResNet
        if padding2 is None:
            self.padding2 = self.padding
        self.stride = stride
        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        # print "init padding = ", padding
        if NAME_RULE == 0:
            new_node.name = "Conv2d(%s, %s)" % (node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "Conv2d"
        else:
            new_node.name = "conv2d"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (node_A.name, node_B.name)
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
        Y = np.empty(y_shape, dtype=X.dtype)

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
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_conv2d(self, X, Filter, padding=0, stride=1):
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

    def profile(self, node, input_vals, output_val, is_static = True):

        assert len(input_vals) == 2
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape) + \
                                         get_array_memory(input_vals[1].shape)
            # output memory
            node.profiler.output_memory = get_array_memory(output_val.shape)
            # TODO
            # no workspace
            node.profiler.workspace_memory = 0
            # execute time
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            # import time
            # start = time.time()
            from ..gpu_links import CuDNN_conv2d
            CuDNN_conv2d(input_vals[0], input_vals[1],
                         output_val, self.padding, self.padding2, self.stride, None, node.profiler)
            # print("time.time: {} ms".format((time.time() - start) * 1000))
            # node.profiler.time = time.time() - start

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 2
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlConv2d']:
                from ..cpu_links import conv2d as cpu_conv2d
                from ..ndarray import numpyasdlarrayhandle
                input_x = numpyasdlarrayhandle(input_vals[0])
                input_f = numpyasdlarrayhandle(input_vals[1])
                output = numpyasdlarrayhandle(output_val)
                cpu_conv2d(input_x, input_f, output, self.padding, self.stride)
            else:
                output_val[:] = self.np_conv2d(
                    input_vals[0], input_vals[1], self.padding, self.stride)
        else:
            from ..gpu_links import CuDNN_conv2d
            CuDNN_conv2d(input_vals[0], input_vals[1],
                         output_val, self.padding, self.padding2, self.stride, stream_handle, None)

    def gradient(self, node, output_grad):
        return [conv2d_gradient_of_data_op(node.inputs[1], output_grad, self.padding, self.padding2, self.stride, self.For_ResNet),\
                conv2d_gradient_of_filter_op(node.inputs[0], output_grad, self.padding, self.padding2, self.stride)]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        # print "infer padding = ",self.padding
        N, _, H, W = input_shapes[0]
        f_O, _, f_H, f_W = input_shapes[1]
        padding = self.padding
        padding2 = self.padding2
        stride = self.stride
        filter_H = input_shapes[1][2]
        filter_W = input_shapes[1][3]
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding2 - filter_W) / stride + 1
        # print "conv2d_shape"
        # print(N, f_O, out_H, out_W)
        return (N, f_O, out_H, out_W)


class Conv2d_Gradient_of_DataOp(Op):
    # nodeA : filter  nodeB : Y_gradient
    def __call__(self, node_A, node_B, padding=0, padding2 = None, stride=1, For_ResNet = False):
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]

        self.padding = padding
        self.padding2 = padding2
        self.stride = stride
        self.For_ResNet = For_ResNet

        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()
        if NAME_RULE == 0:
            new_node.name = "Conv2d_Gradient_of_DataOp(%s, %s)" % (
                node_A.name, node_B.name)
        elif NAME_RULE == 1:
            new_node.name = "Conv2d_Gradient_of_DataOp"
        else:
            new_node.name = "Conv2d_Gradient_of_DataOp"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (node_A.name, node_B.name)
        return new_node

    def im2col_transpose(self, N, C, H, W, filter_H, filter_W, Y, padding, stride):
        assert (H + 2 * padding - filter_H) % stride == 0
        assert (W + 2 * padding - filter_W) % stride == 0
        out_H = (H + 2 * padding - filter_H) / stride + 1
        out_W = (W + 2 * padding - filter_W) / stride + 1
        _, y_row_size, y_col_size = Y.shape

        der_X_shape = (N, C, H, W)
        der_X = np.zeros(der_X_shape, dtype=Y.dtype)

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
                                der_X[batch_index, c, y,
                                      x] += Y[batch_index, row_idx, col_index]
                            row_idx += 1
        return der_X

    def np_Conv2dGradient_data(self, X_N, X_C, X_H, X_W, Filter, Y, padding=0, stride=1):
        filter_outChannel, filter_inChannel, filter_H, filter_W = Filter.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        F_filter = Filter.reshape((filter_outChannel, -1))

        gradient_im2col_XX = np.matmul(F_filter.T, YY)
        gradient_X = self.im2col_transpose(
            X_N, X_C, X_H, X_W, filter_H, filter_W, gradient_im2col_XX, padding, stride)    # gradient of x
        return gradient_X

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
            # import time
            # start = time.time()
            from ..gpu_links import CuDNN_conv2d_gradient_of_data
            CuDNN_conv2d_gradient_of_data(
                input_vals[0], input_vals[1], output_val, padding=self.padding, padding2=self.padding2, stride=self.stride, stream = None, profiler = node.profiler)
            # node.profiler.time = time.time() - start

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):

        assert len(input_vals) == 2
        N = input_vals[1].shape[0]
        C = input_vals[0].shape[1]
        H = (input_vals[1].shape[2] - 1) * self.stride + \
            input_vals[0].shape[2] - 2 * self.padding
        W = (input_vals[1].shape[3] - 1) * self.stride + \
            input_vals[0].shape[3] - 2 * self.padding
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlConv2d_Gradient_of_Data']:
                from ..cpu_links import conv2d_gradient_of_data as cpu_conv2d_gradient_of_data
                from ..ndarray import numpyasdlarrayhandle
                input_f = numpyasdlarrayhandle(input_vals[0])
                gradient_y = numpyasdlarrayhandle(input_vals[1])
                gradient_x = numpyasdlarrayhandle(output_val)
                cpu_conv2d_gradient_of_data(input_f, gradient_y, gradient_x, self.padding, self.stride)
            else:
                output_val[:] = self.np_Conv2dGradient_data(
                    N, C, H, W, input_vals[0], input_vals[1], padding=self.padding, stride=self.stride)
        else:
            from ..gpu_links import CuDNN_conv2d_gradient_of_data
            CuDNN_conv2d_gradient_of_data(
                input_vals[0], input_vals[1], output_val, padding=self.padding, padding2 = self.padding2, stride=self.stride, stream = stream_handle, profiler = None)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        # print self.For_ResNet
        assert len(input_shapes) == 2
        N = input_shapes[1][0]
        C = input_shapes[0][1]
        H = (input_shapes[1][2] - 1) * self.stride + \
            input_shapes[0][2] - 2 * self.padding + (1 if self.For_ResNet and self.stride == 2 else 0)
        W = (input_shapes[1][3] - 1) * self.stride + \
            input_shapes[0][3] - 2 * self.padding2 + (1 if self.For_ResNet and self.stride == 2 else 0)
        return (N, C, H, W)


class Conv2d_Gradient_of_FilterOp(Op):
    # nodeA : input_x  nodeB : gradient_Y
    def __call__(self, input_X, gradient_Y, padding=0, padding2=None, stride=1):
        new_node = Op.__call__(self)
        new_node.inputs = [input_X, gradient_Y]

        self.padding = padding
        if padding2 is None:
            self.padding2 = self.padding
        else:
            self.padding2 = padding2
        self.stride = stride

        new_node.profiler = None
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()

        if NAME_RULE == 0:
            new_node.name = "Conv2d_Gradient_of_FilterOp(%s, %s)" % (
                input_X.name, gradient_Y.name)
        elif NAME_RULE == 1:
            new_node.name = "Conv2d_Gradient_of_FilterOp"
        else:
            new_node.name = "Conv2d_Gradient_of_FilterOp"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (input_X.name, gradient_Y.name)
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
        Y = np.empty(y_shape, dtype=X.dtype)

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
                                Y[batch_index, row_idx,
                                    col_index] = X[batch_index, c, y, x]
                            row_idx += 1
        return Y

    def np_Conv2dGradient_Filter(self, filter_outChannel, filter_inChannel, filter_H, filter_W, X, Y, padding=0, stride=1):
        """Implement a conv2d_transpose as a matrix multiply after im2col."""
        X_N, X_C, X_H, X_W = X.shape
        Y_N, Y_C, Y_H, Y_W = Y.shape
        YY = Y.reshape((Y_N, Y_C, Y_H * Y_W))    # transformed to im2col Y
        # XX = X.reshape((X_N, X_C, X_W * X_H))   # transformed to im2col X
        im2col_XX = self.im2col(X, filter_H, filter_W, padding, stride)
        gradient_filter = np.zeros(shape=(
            filter_outChannel, filter_inChannel * filter_H * filter_W), dtype=Y.dtype)

        for i in range(X_N):
            gradient_filter += np.matmul(YY[i], im2col_XX[i].T)
        gradient_filter = gradient_filter.reshape(
            (filter_outChannel, filter_inChannel, filter_H, filter_W))

        return gradient_filter
        # out_H = (H + 2 * padding - filter_H) / stride + 1
        # out_W = (W + 2 * padding - filter_W) / stride + 1

    def profile(self, node, input_vals, output_val, is_static = True):

        assert len(input_vals) == 2
        if is_static:
            # input memory
            node.profiler.input_memory = get_array_memory(input_vals[0].shape)
                                        #  get_array_memory(input_vals[1].shape)
            # output memory
            node.profiler.output_memory = get_array_memory(output_val.shape)
            # no workspace
            node.profiler.workspace_memory = 0
            # execute time
            node.profiler.time = node.profiler.output_memory / 4 * profiler.FLOPS_PER_SECOND
        else:
            # import time
            # start = time.time()
            from ..gpu_links import CuDNN_conv2d_gradient_of_filter
            CuDNN_conv2d_gradient_of_filter(
                input_vals[0], input_vals[1], output_val, padding=self.padding, padding2=self.padding2, stride=self.stride, stream = None, profiler = node.profiler)
            # node.profiler.time = time.time() - start

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert len(input_vals) == 2
        f_N = input_vals[1].shape[1]
        f_C = input_vals[0].shape[1]
        f_H = input_vals[1].shape[2] + 2 * self.padding - \
            (input_vals[1].shape[2] - 1) * self.stride
        f_W = input_vals[1].shape[3] + 2 * self.padding - \
            (input_vals[1].shape[3] - 1) * self.stride
        if use_numpy:
            from .._base import DNNL_LIB
            if DNNL_LIB['DnnlConv2d_Gradient_of_Filter']:
                from ..cpu_links import conv2d_gradient_of_filter as cpu_conv2d_gradient_of_filter
                from ..ndarray import numpyasdlarrayhandle
                input_x = numpyasdlarrayhandle(input_vals[0])
                gradient_y = numpyasdlarrayhandle(input_vals[1])
                gradient_f = numpyasdlarrayhandle(output_val)
                cpu_conv2d_gradient_of_filter(input_x, gradient_y, gradient_f, self.padding, self.stride)
            else:
                output_val[:] = self.np_Conv2dGradient_Filter(
                    f_N, f_C, f_H, f_W, input_vals[0], input_vals[1], padding=self.padding, stride=self.stride)
        else:
            from ..gpu_links import CuDNN_conv2d_gradient_of_filter
            CuDNN_conv2d_gradient_of_filter(
                input_vals[0], input_vals[1], output_val, padding=self.padding, padding2=self.padding2, stride=self.stride, stream = stream_handle, profiler = None)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """TODO: Your code here"""
        assert len(input_shapes) == 2
        f_N = input_shapes[1][1]
        f_C = input_shapes[0][1]
        f_H = input_shapes[0][2] + 2 * self.padding - \
            (input_shapes[1][2] - 1) * self.stride
        f_W = input_shapes[0][3] + 2 * self.padding2 - \
            (input_shapes[1][3] - 1) * self.stride

        return (f_N, f_C, f_H, f_W)


def conv2d_op(node_A, node_B, padding=0, padding2=None, stride=1, For_ResNet = False):
    """Conv2d node.

    Parameters:
    ----
    node_A : Node
        Input data node.
    node_B : Node
        Input filter node.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2dOp()(node_A, node_B, padding, padding2, stride, For_ResNet)


def conv2d_gradient_of_data_op(node_A, node_B, padding=0, padding2 = None,stride=1, For_ResNet = False):
    """Gradient node of data of conv2d.

    Parameters:
    ----
    node_A : Node
        Filter node.
    node_B : Node
        Previous gradient node.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2d_Gradient_of_DataOp()(node_A, node_B, padding, padding2, stride, For_ResNet)


def conv2d_gradient_of_filter_op(input_X, gradient_Y, padding=0, padding2=None, stride=1):
    """Gradient node of filters of conv2d.

    Parameters:
    ----
    input_X :
        Input data of conv2d.
    gradient_Y :
        Gradient array.
    padding :
        Padding size.
    stride :
        Stride size.

    Returns:
    ----
    A new Node instance created by Op.

    """
    return Conv2d_Gradient_of_FilterOp()(input_X, gradient_Y, padding, padding2, stride)
