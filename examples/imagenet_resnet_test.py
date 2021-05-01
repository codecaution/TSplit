import numpy as np
from athena import ndarray
from athena import gpu_ops as ad
from athena.microopOptimizer import microopOptimizer
from athena.microopPlanner import microopPlanner
import time
import argparse
executor_ctx = ndarray.gpu(0)
variable_list = []
val_list = []

rand = np.random.RandomState(seed=123)


def get_variable(name, size):
    global variable_list, val_list
    x = ad.Variable(name=name)
    x_val = rand.normal(scale=0.1, size=size)
    x_val = ndarray.array(x_val, ctx=executor_ctx)
    variable_list.append(x)
    val_list.append(x_val)
    return x


def conv2d_1_1(x, in_channel, out_channel, stride=1, padding=1, name=''):
    x = ad.conv2d_op(x, get_variable(name + '_weight', (out_channel, in_channel, 1, 1)), stride=stride, padding=padding, For_ResNet=True)
    return x


def conv2d_3_3(x, in_channel, out_channel, stride=1, padding=1, name=''):
    x = ad.conv2d_op(x, get_variable(name + '_weight', (out_channel, in_channel, 3, 3)), stride=stride, padding=padding, For_ResNet=True)
    return x

def conv2d_7_7(x, in_channel, out_channel, stride=1, padding=1, name=''):
    x = ad.conv2d_op(x, get_variable(name + '_weight', (out_channel, in_channel, 7, 7)), stride=stride, padding=padding, For_ResNet=True)
    return x

def batch_norm_with_relu(x, hidden, name):
    x = ad.batch_normalization_op(x, get_variable(name + '_scale', (1, hidden, 1, 1)),
                                  get_variable(name + '_bias', (1, hidden, 1, 1)))
    x = ad.relu_op(x)
    return x


def resnet_block_large(x, in_channel, out_channel, num_blocks, is_first=False, name=''):
    if is_first:
        indentity = conv2d_1_1(x, in_channel, out_channel, stride=1, padding=0, name=name + '_conv0')
        indentity = batch_norm_with_relu(indentity, out_channel, name + '_bn0')
        x = conv2d_1_1(x, in_channel, out_channel / 4, stride=1, padding=0, name=name + '_conv1')
        x = batch_norm_with_relu(x, out_channel / 4, name + '_bn1')
        x = conv2d_3_3(x, out_channel / 4, out_channel / 4, stride=1, padding=1, name=name + '_conv2')
        x = batch_norm_with_relu(x, out_channel / 4, name + '_bn2')
        x = conv2d_1_1(x, out_channel / 4, out_channel, stride=1, padding=0, name=name + '_conv3')
        x = batch_norm_with_relu(x, out_channel, name + 'bn_3')
        x = x + indentity
    else:
        identity = conv2d_1_1(x, in_channel, out_channel, stride=2, padding=0, name=name + '_conv0')
        identity = batch_norm_with_relu(identity, out_channel, name + '_bn0')
        x = conv2d_1_1(x, in_channel, out_channel / 4, stride=1, padding=0, name=name + '_conv1')
        x = batch_norm_with_relu(x, out_channel / 4, name + '_bn1')
        x = conv2d_3_3(x, out_channel / 4 , out_channel / 4, stride=2, padding=1, name=name + '_conv2')
        x = batch_norm_with_relu(x, out_channel / 4, name + '_bn2')
        x = conv2d_1_1(x, out_channel / 4, out_channel, stride=1, padding=0, name=name + '_conv3')
        x = batch_norm_with_relu(x, out_channel, name + '_bn3')
        x = x + identity
    for i in range(1, num_blocks):
        identity = x
        x = conv2d_1_1(x, out_channel, out_channel / 4, stride=1, padding=0, name=name + '_conv%d' % (3 * i + 1))
        x = batch_norm_with_relu(x, out_channel / 4, name + '_bn%d' % (3 * i + 1))
        x = conv2d_3_3(x, out_channel / 4, out_channel / 4, stride=1, padding=1, name=name + '_conv%d' % (3 * i + 2))
        x = batch_norm_with_relu(x, out_channel / 4, name + '_bn%d' % (3 * i + 2))
        x = conv2d_1_1(x, out_channel / 4, out_channel, stride=1, padding=0, name=name + '_conv%d' % (3 * i + 3))
        x = batch_norm_with_relu(x, out_channel, name + '_bn%d' % (3 * i + 3))
        x = x + identity

    return x

def fc(x, shape, name):
    x = ad.matmul_op(x, get_variable(name + '_weight', shape))
    return x


def resnet_model(x, y_, num_layers=18):
    '''
    ResNet model, for CIFAR10 dataset.
    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, C, H, W)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
        num_layers: 18 or 34
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    base_size = 64

    x = conv2d_7_7(x, 3, base_size, stride=2, padding=3, name='resnet_initial_conv')
    x = batch_norm_with_relu(x, base_size, 'resnet_initial_bn')
    x = ad.max_pool2d_op(x, 3, 3, stride=2, padding=1)

    if num_layers == 50:
        # print("Building ResNet-50 model...")
        x = resnet_block_large(x, base_size, 4 * 64, num_blocks=3, is_first=True, name='resnet_block1')
        x = resnet_block_large(x, 4 * 64, 4 * 128, num_blocks=4, is_first=False, name='resnet_block2')
        x = resnet_block_large(x, 4 * 128, 4 * 256, num_blocks=6, is_first=False, name='resnet_block3')
        x = resnet_block_large(x, 4 * 256, 4 * 512, num_blocks=3, is_first=False, name='resnet_block4')
    elif num_layers == 101:
        # print("Building ResNet-101 model...")
        x = resnet_block_large(x, base_size, 4 * 64, num_blocks=3, is_first=True, name='resnet_block1')
        x = resnet_block_large(x, 4 * 64, 4 * 128, num_blocks=4, is_first=False, name='resnet_block2')
        x = resnet_block_large(x, 4 * 128, 4 * 256, num_blocks=23, is_first=False, name='resnet_block3')
        x = resnet_block_large(x, 4 * 256, 4 * 512, num_blocks=3, is_first=False, name='resnet_block4')
    else:
        assert False, "Number of layers should be 18, 34, 50 or 101 !"

    x = ad.avg_pool2d_op(x, 7, 7, padding=0, stride=7)
    x = ad.array_reshape_op(x, (batch_size, -1))
    y = fc(x, (512 * 4, 1000), name='resnet_final_fc')
    # here we don't use cudnn for softmax crossentropy to avoid overflows
    loss = ad.softmaxcrossentropy_op(y, y_)
    return loss, y


def resnet(batch_size, num_layers, policy = "None"):
    global variable_list, val_list
    variable_list = []
    val_list = []

    X = ad.Variable(name='X')
    X_val = np.empty(shape=(batch_size, 3, 224, 224), dtype=np.float32)
    # X_val = ndarray.array(X_val, ctx=executor_ctx)
    y_ = ad.Variable(name='y_')
    y_val = np.empty(shape=(batch_size, 1000), dtype=np.float32)
    # y_val = ndarray.array(y_val, ctx=executor_ctx)

    loss, y = resnet_model(X, y_, num_layers)
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
    for i in range(2):
        if i == 1:
            start = time.time()
        grad_val_list = executor.run(feed_dict)

    end = time.time()
    return (end - start) / 1

if __name__ == "__main__":
    batch_size = 360
    layer = 101
    execution_time = resnet(batch_size, layer, policy = "superneurons")
    print(execution_time)

