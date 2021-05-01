from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad
import numpy as np
import argparse
import six.moves.cPickle as pickle
import gzip
import os
import pdb
import time
import logging
from athena import gpu_memory_manager

channel_axis = 1

variable_list = []
val_list = []

rand = np.random.RandomState(seed=123)

def load_cifar10_data(directory):
    images, labels = [], []
    for filename in ['%s/data_batch_%d' % (directory, j) for j in range(1, 6)]:
        with open(filename, 'rb') as fo:
            cifar10 = pickle.load(fo)
        for i in range(len(cifar10[b"labels"])):
            # image = np.reshape(cifar10[b"data"][i], (3, 32, 32))
            image = cifar10[b"data"][i]
            image = image.astype(float)
            images.append(image)
        labels += cifar10[b"labels"]
    images = np.array(images, dtype='float')
    labels = np.array(labels, dtype='int')
    train_images, train_labels = images, labels

    images, labels = [], []
    for filename in ['%s/test_batch' % (directory)]:
        with open(filename, 'rb') as fo:
            cifar10 = pickle.load(fo)
        for i in range(len(cifar10[b"labels"])):
            # image = np.reshape(cifar10[b"data"][i], (3, 32, 32))
            image = cifar10[b"data"][i]
            image = image.astype(float)
            images.append(image)
        labels += cifar10[b"labels"]
    images = np.array(images, dtype='float')
    labels = np.array(labels, dtype='int')
    test_images, test_labels = images, labels
    print
    train_images.shape
    return train_images, train_labels, test_images, test_labels


def convert_to_one_hot(vals, max_val=0):
    """Helper method to convert label array to one-hot array."""
    if max_val == 0:
        max_val = vals.max() + 1
    one_hot_vals = np.zeros((vals.size, max_val))
    one_hot_vals[np.arange(vals.size), vals] = 1
    return one_hot_vals


def sgd_update_gpu(param, grad_param, learning_rate, swap=False):
    """Helper GPU SGD update method. Avoids copying NDArray to cpu."""
    if not swap:
        assert isinstance(param, ndarray.NDArray)
        assert isinstance(grad_param, ndarray.NDArray)
    if swap:
        param = param - learning_rate * grad_param
    else:
        gpu_op.matrix_elementwise_multiply_by_const(
            grad_param, -learning_rate, grad_param)
        gpu_op.matrix_elementwise_add(param, grad_param, param)

def get_variable(name, size):
    global variable_list, val_list
    x = ad.Variable(name=name)
    x_val = rand.normal(scale=0.1, size=size)
    x_val = ndarray.array(x_val, ctx=ndarray.gpu(0))
    variable_list.append(x)
    val_list.append(x_val)
    return x

def conv2d_bn(x, nb_filter, ni_filter, num_row, num_col, stride = 1, padding = 'same'):
    if padding != 'same':
        x = ad.conv2d_op(x, get_variable('W', (nb_filter, ni_filter, num_row, num_col)), stride=stride) #oihw
    else:
        # x = ad.pad_op(x, [[0, 0], [0, 0], [0, num_row - stride], [0, num_col - stride]])
        x = ad.conv2d_op(x, get_variable('W', (nb_filter, ni_filter, num_row, num_col)),
                         (num_row - stride) / 2, (num_col - stride) / 2, stride=stride) #oihw
    x = ad.batch_normalization_op(
        x,get_variable('b_scale', (1, nb_filter, 1, 1)), get_variable('b_bias', (1, nb_filter, 1, 1)))
    x = ad.relu_op(x)
    return x

def block_inception_a(input, ni_filter):
    branch_0 = conv2d_bn(input, 96, ni_filter, 1, 1)

    branch_1 = conv2d_bn(input, 64, ni_filter, 1, 1)
    branch_1 = conv2d_bn(branch_1, 96, 64, 3, 3)

    branch_2 = conv2d_bn(input, 64, ni_filter, 1, 1)
    branch_2 = conv2d_bn(branch_2, 96, 64, 3, 3)
    branch_2 = conv2d_bn(branch_2, 96, 96, 3, 3)

    branch_3 = ad.avg_pool2d_op(input, 3, 3, 1, 1)
    branch_3 = conv2d_bn(branch_3, 96, ni_filter, 1, 1)

    x = ad.concat_op(branch_0, branch_1, axis=1)
    x = ad.concat_op(x, branch_2, axis=1)
    x = ad.concat_op(x, branch_3, axis=1)
    return x

def block_reduction_a(input, ni_filter):
    branch_0 = conv2d_bn(input, 384, ni_filter, 3, 3, stride=2, padding='valid')

    branch_1 = conv2d_bn(input, 192, ni_filter, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 192, 3, 3)
    branch_1 = conv2d_bn(branch_1, 256, 224, 3, 3, stride=2, padding='valid')

    branch_2 = ad.max_pool2d_op(input, 3, 3, 0, 2)

    x = ad.concat_op(branch_0, branch_1, axis=1)
    x = ad.concat_op(x, branch_2, axis=1)
    return x

def block_inception_b(input, ni_filter):
    branch_0 = conv2d_bn(input, 384, ni_filter, 1, 1)

    branch_1 = conv2d_bn(input, 192, ni_filter, 1, 1)
    branch_1 = conv2d_bn(branch_1, 224, 192, 1, 7)
    branch_1 = conv2d_bn(branch_1, 256, 224, 7, 1)

    branch_2 = conv2d_bn(input, 192, ni_filter, 1, 1)
    branch_2 = conv2d_bn(branch_2, 192, 192, 7, 1)
    branch_2 = conv2d_bn(branch_2, 224, 192, 1, 7)
    branch_2 = conv2d_bn(branch_2, 224, 224, 7, 1)
    branch_2 = conv2d_bn(branch_2, 256, 224, 1, 7)

    branch_3 = ad.avg_pool2d_op(input, 3, 3, 1, 1)
    branch_3 = conv2d_bn(branch_3, 128, ni_filter, 1, 1)

    x = ad.concat_op(branch_0, branch_1, axis=1)
    x = ad.concat_op(x, branch_2, axis=1)
    x = ad.concat_op(x, branch_3, axis=1)
    return x

def block_reduction_b(input, ni_filter):
    branch_0 = conv2d_bn(input, 192, ni_filter, 1, 1)
    branch_0 = conv2d_bn(branch_0, 192, 192, 3, 3, stride=2, padding='valid')

    branch_1 = conv2d_bn(input, 256, ni_filter, 1, 1)
    branch_1 = conv2d_bn(branch_1, 256, 256, 1, 7)
    branch_1 = conv2d_bn(branch_1, 320, 256, 7, 1)
    branch_1 = conv2d_bn(branch_1, 320, 320, 3, 3, stride=2, padding='valid')

    branch_2 = ad.max_pool2d_op(input, 3, 3, 0, 2)

    x = ad.concat_op(branch_0, branch_1, axis=1)
    x = ad.concat_op(x, branch_2, axis=1)
    return x

def block_inception_c(input, ni_filter):
    branch_0 = conv2d_bn(input, 256, ni_filter, 1, 1)

    branch_1 = conv2d_bn(input, 384, ni_filter, 1, 1)
    branch_10 = conv2d_bn(branch_1, 256, 384, 1, 3)
    branch_11 = conv2d_bn(branch_1, 256, 384, 3, 1)
    branch_1 = ad.concat_op(branch_10, branch_11, axis=1)

    branch_2 = conv2d_bn(input, 384, ni_filter, 1, 1)
    branch_2 = conv2d_bn(branch_2, 448, 384, 3, 1)
    branch_2 = conv2d_bn(branch_2, 512, 448, 1, 3)
    branch_20 = conv2d_bn(branch_2, 256, 512, 1, 3)
    branch_21 = conv2d_bn(branch_2, 256, 512, 3, 1)
    branch_2 = ad.concat_op(branch_20, branch_21, axis=1)

    branch_3 =  ad.avg_pool2d_op(input, 3, 3, 1, 1)
    branch_3 = conv2d_bn(branch_3, 256, ni_filter, 1, 1)

    x = ad.concat_op(branch_0, branch_1, axis=1)
    x = ad.concat_op(x, branch_2, axis=1)
    x = ad.concat_op(x, branch_3, axis=1)
    return x

def inception_v4_base(input, ni_filter):
    net = conv2d_bn(input, 32, ni_filter, 3, 3, stride=2, padding='valid')
    net = conv2d_bn(net, 32, 32, 3, 3, padding='valid')
    net = conv2d_bn(net, 64, 32, 3, 3)

    branch_0 = ad.max_pool2d_op(net, 3, 3, 0, 2)

    branch_1 = conv2d_bn(net, 96, 64, 3, 3, stride=2, padding='valid')

    net = ad.concat_op(branch_0, branch_1, axis=1)

    branch_0 = conv2d_bn(net, 64, 160, 1, 1)
    branch_0 = conv2d_bn(branch_0, 96, 64, 3, 3, padding='valid')

    branch_1 = conv2d_bn(net, 64, 160, 1, 1)
    branch_1 = conv2d_bn(branch_1, 64, 64, 1, 7)
    branch_1 = conv2d_bn(branch_1, 64, 64, 7, 1)
    branch_1 = conv2d_bn(branch_1, 96, 64, 3, 3, padding='valid')

    net = ad.concat_op(branch_0, branch_1, axis=1)

    branch_0 = conv2d_bn(net, 192, 192, 3, 3, stride=2, padding='valid')
    branch_1 = ad.max_pool2d_op(net, 3, 3, 0, 2)

    net = ad.concat_op(branch_0, branch_1, axis=1)

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in range(4):
        net = block_inception_a(net, 384)

    # 35 x 35 x 384
    # Reduction-A block
    net = block_reduction_a(net, 384)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in range(7):
        net = block_inception_b(net, 1024)

    # 17 x 17 x 1024
    # Reduction-B block
    net = block_reduction_b(net, 1024)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in range(3):
        net = block_inception_c(net, 1536)

    return net

def inception_v4(batch_size = 32, policy = "None"):
    global variable_list, val_list
    executor_ctx = ndarray.gpu(0)
    variable_list = []
    val_list = []

    X = ad.Variable(name='X')
    X_val = np.empty(shape=(batch_size, 3, 299, 299), dtype=np.float32)
    # X_val = ndarray.array(X_val, ctx=ndarray.gpu(0))
    y_ = ad.Variable(name='y_')
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    # y_val = ndarray.array(y_val, ctx=ndarray.gpu(0))
    model = inception_v4_base(X, 3)
    model = ad.avg_pool2d_op(model, 8, 8, 0, 1)
    model = ad.array_reshape_op(model, (batch_size, -1))
    model = ad.matmul_op(model, get_variable('W', (1536, 10)))
    y = model + ad.broadcastto_op(get_variable('b', (10)), (batch_size, 10))

    loss = ad.softmaxcrossentropy_op(y, y_)
    grad_list = ad.gradients(loss, variable_list)

    if policy == "None" or policy == "base":
        athena_exec = ad.Executor
    elif policy == "vdnnconv" or policy == "vdnnall":
        athena_exec = ad.vdnnExecutor
    elif policy == "superneurons":
        athena_exec = ad.superNeuronsExecutor
    elif policy == "recompute_memory" or policy == "recompute_speed":
        athena_exec = ad.recomputeExecutor
    elif policy == "simulator":
        athena_exec = microopOptimizer
    elif policy == "profiler":
        athena_exec = ad.profileExecutor
    elif policy == "planner":
        athena_exec = microopPlanner
    elif policy == "tsplit":
        athena_exec = ad.microopExecutor
    else:
        raise NotImplementedError

    if policy == "vdnnconv":
        executor = athena_exec([loss] + grad_list + [y], ctx=executor_ctx, policy = "conv")
    elif policy == "vdnnall": 
        executor = athena_exec([loss] + grad_list + [y], ctx=executor_ctx, policy = "all")
    elif policy == "recompute_memory":
        executor = athena_exec([loss] + grad_list + [y], ctx=executor_ctx, policy = "memory")
    elif policy == "recompute_speed":
        executor = athena_exec([loss] + grad_list + [y], ctx=executor_ctx, policy = "speed")
    else:
        executor = athena_exec([loss] + grad_list + [y], ctx=executor_ctx)

    feed_dict = dict()
    feed_dict[X] = X_val
    feed_dict[y_] = y_val
    for i in range(len(variable_list)):
        feed_dict[variable_list[i]] = val_list[i]
    for i in range(3):
        if i == 1:
            start = time.time()
        grad_val_list = executor.run(feed_dict)
    end = time.time()

    return (end - start) / 2

if __name__ == "__main__":

    # batch_size = 1372
    # execution_time = inception_v4(batch_size, policy = "tsplit")
    # print("execution time:", execution_time)
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-p','--policy', default='None')
    args = parser.parse_args()

    policy = args.policy
    output_file_name = "/home/xiaonan/microop/Athena/exp_12G/" + "inceptionV4/" + policy + "_batchsize_with_time.txt"
    output_file = open(output_file_name, "a+", buffering=1)
    output_file.write("Policy: {}, on InceprionV4\n".format(policy))
    # print(policy)
    for batch_size in range(32, 2000, 32):
        execution_time = inception_v4(batch_size, policy = policy)
        print("Batch size: {} , time: {} s\n".format(batch_size, execution_time))
        output_file.write("Batch size: {} , time: {} s\n".format(batch_size, execution_time))
    output_file.close()