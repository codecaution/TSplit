from __future__ import absolute_import
import os
import ctypes
from .._base import _LIB
import numpy as np
from .. import ndarray as _nd

def matrix_multiply(matA, transposeA, matB, transposeB, matC):
    assert isinstance(matA, _nd.DLArray)
    assert isinstance(matB, _nd.DLArray)
    assert isinstance(matC, _nd.DLArray)
    _LIB.DnnlMatrixMultiply(ctypes.byref(matA), transposeA, ctypes.byref(matB), transposeB, ctypes.byref(matC))

def matrix_elementwise_multiply_by_const(mat, val, output):
    assert isinstance(mat, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlMatrixElementwiseMultiplyByConst(ctypes.byref(mat), ctypes.c_float(val), ctypes.byref(output))

def matrix_elementwise_multiply(matA, matB, output):
    assert isinstance(matA, _nd.DLArray)
    assert isinstance(matB, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlMatrixElementwiseMultiply(ctypes.byref(matA), ctypes.byref(matB), ctypes.byref(output))

def matrix_elementwise_add_by_const(mat, val, output):
    assert isinstance(mat, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlMatrixElementwiseAddByConst(ctypes.byref(mat), ctypes.c_float(val), ctypes.byref(output))

def matrix_elementwise_add(matA, matB, output):
    assert isinstance(matA, _nd.DLArray)
    assert isinstance(matB, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlMatrixElementwiseAdd(ctypes.byref(matA), ctypes.byref(matB), ctypes.byref(output))

def matrix_elementwise_divide_by_const(mat, val, output):
    assert isinstance(mat, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlMatrixElementwiseDivideByConst(ctypes.byref(mat), ctypes.c_float(val), ctypes.byref(output))

def matrix_elementwise_divide(matA, matB, output):
    assert isinstance(matA, _nd.DLArray)
    assert isinstance(matB, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlMatrixElementwiseDivide(ctypes.byref(matA), ctypes.byref(matB), ctypes.byref(output))

def broadcast_to(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.cpu_BroadcastTo(ctypes.byref(in_arr), ctypes.byref(out_arr))

def reduce_sum_axis_zero(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.cpu_ReduceSumAxisZero(ctypes.byref(in_arr), ctypes.byref(out_arr))

def array_set(output, value):
    assert isinstance(output, _nd.DLArray)
    _LIB.cpu_ArraySet(ctypes.byref(output), ctypes.c_float(value))

def reshape(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.cpu_Reshape(ctypes.byref(in_arr), ctypes.byref(out_arr))

def softmax(mat, output):
    assert isinstance(mat, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlSoftmax(ctypes.byref(mat), ctypes.byref(output))

def softmax_crossentropy(matA, matB, output):
    assert isinstance(matA, _nd.DLArray)
    assert isinstance(matB, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlSoftmaxCrossEntropy(ctypes.byref(matA), ctypes.byref(matB), ctypes.byref(output))

def sqrt(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.DnnlSqrt(ctypes.byref(in_arr), ctypes.byref(out_arr))

def rsqrt(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.DnnlReciprocalSqrt(ctypes.byref(in_arr), ctypes.byref(out_arr))

def tanh(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.DnnlTanh(ctypes.byref(in_arr), ctypes.byref(out_arr))

def opposite(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.DnnlOpposite(ctypes.byref(in_arr), ctypes.byref(out_arr))

def sigmoid(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.DnnlSigmoid(ctypes.byref(in_arr), ctypes.byref(out_arr))

def conv2d(input_x, input_f, output, padding=0, stride=1):
    assert isinstance(input_x, _nd.DLArray)
    assert isinstance(input_f, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlConv2d(ctypes.byref(input_x), ctypes.byref(input_f), ctypes.byref(output), ctypes.c_int(padding), ctypes.c_int(stride))

def conv2d_gradient_of_data(input_f, gradient_y, gradient_x, padding=0, stride=1):
    assert isinstance(gradient_y, _nd.DLArray)
    assert isinstance(input_f, _nd.DLArray)
    assert isinstance(gradient_x, _nd.DLArray)
    _LIB.DnnlConv2d_Gradient_of_Data(ctypes.byref(input_f), ctypes.byref(gradient_y), ctypes.byref(gradient_x), ctypes.c_int(padding), ctypes.c_int(stride))

def conv2d_gradient_of_filter(input_x, gradient_y, gradient_f, padding=0, stride=1):
    assert isinstance(gradient_y, _nd.DLArray)
    assert isinstance(input_x, _nd.DLArray)
    assert isinstance(gradient_f, _nd.DLArray)
    _LIB.DnnlConv2d_Gradient_of_Filter(ctypes.byref(input_x), ctypes.byref(gradient_y), ctypes.byref(gradient_f), ctypes.c_int(padding), ctypes.c_int(stride))

def avg_pool(input, H, W, output, padding=0, stride=1):
    assert isinstance(input, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlAvgPool(ctypes.byref(input), ctypes.c_int(H), ctypes.c_int(W), ctypes.byref(output), ctypes.c_int(padding), ctypes.c_int(stride))

def avg_pool_gradient(gradient_Y, H, W, gradient_X, padding=0, stride=1):
    assert isinstance(gradient_Y, _nd.DLArray)
    assert isinstance(gradient_X, _nd.DLArray)
    _LIB.DnnlAvgPool_Gradient(ctypes.byref(gradient_Y), ctypes.c_int(H), ctypes.c_int(W), ctypes.byref(gradient_X), ctypes.c_int(padding), ctypes.c_int(stride))

def max_pool(input, H, W, output, padding=0, stride=1):
    assert isinstance(input, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlMaxPool(ctypes.byref(input), ctypes.c_int(H), ctypes.c_int(W), ctypes.byref(output), ctypes.c_int(padding), ctypes.c_int(stride))

def max_pool_gradient(input, input_grad, H, W, output, padding=0, stride=1):
    assert isinstance(input, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlMaxPool_Gradient(ctypes.byref(input), ctypes.byref(input_grad), ctypes.c_int(H), ctypes.c_int(W), ctypes.byref(output), ctypes.c_int(padding), ctypes.c_int(stride))

def relu(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.DnnlRelu(ctypes.byref(in_arr), ctypes.byref(out_arr))

def relu_gradient(input, in_grad, output):
    assert isinstance(input, _nd.DLArray)
    assert isinstance(in_grad, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlRelu_Gradient(ctypes.byref(input), ctypes.byref(in_grad), ctypes.byref(output))

def batch_norm(input, bn_scale, bn_bias, output, mean, var, momentum = 0.99, eps = 0.01):
    assert isinstance(input, _nd.DLArray)
    assert isinstance(bn_scale, _nd.DLArray)
    assert isinstance(bn_bias, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    assert isinstance(mean, _nd.DLArray)
    assert isinstance(var, _nd.DLArray)
    _LIB.DnnlBatchNorm(ctypes.byref(input), ctypes.byref(bn_scale), ctypes.byref(bn_bias), ctypes.byref(output),
                       ctypes.byref(mean), ctypes.byref(var), ctypes.c_float(momentum), ctypes.c_float(eps))

def batch_norm_gradient(gradient_Y, input_X, bn_scale, bn_bias, gradient_X, gradient_bn_scale, gradient_bn_bias, mean, var, eps=0.01):
    assert isinstance(gradient_Y, _nd.DLArray)
    assert isinstance(input_X, _nd.DLArray)
    assert isinstance(gradient_X, _nd.DLArray)
    assert isinstance(gradient_bn_scale, _nd.DLArray)
    assert isinstance(gradient_bn_bias, _nd.DLArray)
    assert isinstance(bn_scale, _nd.DLArray)
    assert isinstance(bn_bias, _nd.DLArray)
    assert isinstance(mean, _nd.DLArray)
    assert isinstance(var, _nd.DLArray)
    _LIB.DnnlBatchNorm_Gradient(ctypes.byref(gradient_Y), ctypes.byref(input_X), ctypes.byref(bn_scale),
                                ctypes.byref(bn_bias), ctypes.byref(gradient_X), ctypes.byref(gradient_bn_scale),
                                ctypes.byref(gradient_bn_bias), ctypes.byref(mean), ctypes.byref(var), ctypes.c_float(eps))


def concat(input_x, input_y, output, axis=0):
    assert isinstance(input_x, _nd.DLArray)
    assert isinstance(input_y, _nd.DLArray)
    assert isinstance(output, _nd.DLArray)
    _LIB.DnnlConcat(ctypes.byref(input_x), ctypes.byref(input_y), ctypes.byref(output), ctypes.c_int(axis))

def concat_gradient(output_gradient, input_gradient, axis=0,id=0):
    assert isinstance(output_gradient, _nd.DLArray)
    assert isinstance(input_gradient, _nd.DLArray)
    _LIB.cpu_Concat_Gradient(ctypes.byref(output_gradient), ctypes.byref(input_gradient), ctypes.c_int(axis), ctypes.c_int(id))

def Dropout(in_arr, dropout, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.cpu_Dropout(ctypes.byref(in_arr), ctypes.c_float(dropout), ctypes.byref(out_arr))


def Dropout_gradient(in_gradient_y, dropout, out_gradient_x):
    assert isinstance(in_gradient_y, _nd.DLArray)
    assert isinstance(out_gradient_x, _nd.DLArray)
    _LIB.cpu_Dropout_Gradient(ctypes.byref(in_gradient_y), ctypes.c_float(dropout), ctypes.byref(out_gradient_x))

def pad(in_arr, out_arr, paddings, mode='CONSTANT', constant_values=0):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    padding_arr = []
    for i in range(len(paddings)):
        for j in range(len(paddings[0])):
            padding_arr.append(paddings[i][j])
    pad_len = len(padding_arr)
    padding_c_arr = (ctypes.c_int * pad_len)(*padding_arr)
    # padding_arr = (ctypes.c_char_p)(*padding_arr)
    f_type = 3
    if mode == 'CONSTANT':
        f_type = 0
    elif mode == 'REFLECT':
        f_type = 1
    elif mode == 'SYMMETRIC':
        f_type = 2
    assert(f_type <= 2)
    _LIB.cpu_Pad(ctypes.byref(in_arr), ctypes.byref(out_arr), padding_c_arr,
                  ctypes.c_int(pad_len), ctypes.c_int(f_type), ctypes.c_float(constant_values))

def pad_gradient(out_grad_arr, in_grad_arr, paddings, mode="CONSTANT"):
    assert isinstance(out_grad_arr, _nd.DLArray)
    assert isinstance(in_grad_arr, _nd.DLArray)
    padding_arr = []
    for i in range(len(paddings)):
        for j in range(len(paddings[0])):
            padding_arr.append(paddings[i][j])
    pad_len = len(padding_arr)
    padding_c_arr = (ctypes.c_int * pad_len)(*padding_arr)
    # padding_arr = (ctypes.c_char_p)(*padding_arr)
    f_type = 3
    if mode == 'CONSTANT':
        f_type = 0
    elif mode == 'REFLECT':
        f_type = 1
    elif mode == 'SYMMETRIC':
        f_type = 2
    assert (f_type <= 2)
    _LIB.cpu_Pad_Gradient(ctypes.byref(out_grad_arr),
                           ctypes.byref(in_grad_arr), padding_c_arr, ctypes.c_int(pad_len), ctypes.c_int(f_type))


def transpose(in_arr, out_arr):
    assert isinstance(in_arr, _nd.DLArray)
    assert isinstance(out_arr, _nd.DLArray)
    _LIB.cpu_Transpose(ctypes.byref(in_arr), ctypes.byref(out_arr))

def sgd_update(param, grad, lr):
    assert isinstance(param, _nd.DLArray)
    assert isinstance(grad, _nd.DLArray)
    _LIB.cpu_SGDOptimizerUpdate(ctypes.byref(param), ctypes.byref(grad), ctypes.c_float(lr))

def momentum_update(param, grad, velocity, lr, momentum, nesterov):
    assert isinstance(param, _nd.DLArray)
    assert isinstance(grad, _nd.DLArray)
    assert isinstance(velocity, _nd.DLArray)
    _LIB.cpu_MomentumOptimizerUpdate(ctypes.byref(param), ctypes.byref(grad), ctypes.byref(velocity),
                                     ctypes.c_float(lr), ctypes.c_float(momentum), ctypes.c_bool(nesterov))

def adagrad_update(param, grad, accumulation, lr, eps):
    assert isinstance(param, _nd.DLArray)
    assert isinstance(grad, _nd.DLArray)
    assert isinstance(accumulation, _nd.DLArray)
    _LIB.cpu_AdaGradOptimizerUpdate(ctypes.byref(param), ctypes.byref(grad), ctypes.byref(accumulation),
                                    ctypes.c_float(lr), ctypes.c_float(eps))

def adam_update(param, grad, expavg, expavgsq, lr, beta1, beta2, beta1t, beta2t, eps):
    assert isinstance(param, _nd.DLArray)
    assert isinstance(grad, _nd.DLArray)
    assert isinstance(expavg, _nd.DLArray)
    assert isinstance(expavgsq, _nd.DLArray)
    _LIB.cpu_AdamOptimizerUpdate(ctypes.byref(param), ctypes.byref(grad), ctypes.byref(expavg),
                                 ctypes.byref(expavgsq), ctypes.c_float(lr),
                                 ctypes.c_float(beta1), ctypes.c_float(beta2), ctypes.c_float(beta1t),
                                 ctypes.c_float(beta2t), ctypes.c_float(eps))

