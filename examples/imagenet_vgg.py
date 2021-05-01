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

def conv_bn_relu(x, in_channel, out_channel, name):
    weight = get_variable(name=name+'_weight', size=(out_channel, in_channel, 3, 3))
    bn_scale = get_variable(name=name+'_bn_scale', size=(1, out_channel, 1, 1))
    bn_bias = get_variable(name=name+'_bn_bias', size=(1, out_channel, 1, 1))
    
    conv = ad.conv2d_op(x, weight, padding=1, stride=1)
    bn = ad.batch_normalization_op(conv, bn_scale, bn_bias)
    act = ad.relu_op(bn)
    return act

def vgg_2block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = ad.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x

def vgg_3block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer3')
    x = ad.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x

def vgg_4block(x, in_channel, out_channel, name):
    x = conv_bn_relu(x, in_channel, out_channel, name=name+'_layer1')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer2')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer3')
    x = conv_bn_relu(x, out_channel, out_channel, name=name+'_layer4')
    x = ad.max_pool2d_op(x, kernel_H=2, kernel_W=2, padding=0, stride=2)
    return x

def vgg_fc(x, in_feat, out_feat, name):
    weight = get_variable(name=name+'_weight', size=(in_feat, out_feat))
    x = ad.matmul_op(x, weight)
    return x

def vgg_model(x, y_, num_layers):
    '''
    VGG model, for CIFAR10 dataset.
    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, C, H, W)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
        num_layers: 16 or 19
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''
    
    if num_layers == 16:
        print('Building VGG-16 model...')
        x = vgg_2block(x,   3,  64, 'vgg_block1')
        x = vgg_2block(x,  64, 128, 'vgg_block2')
        x = vgg_3block(x, 128, 256, 'vgg_block3')
        x = vgg_3block(x, 256, 512, 'vgg_block4')
        x = vgg_3block(x, 512, 512, 'vgg_block5')
        
    elif num_layers == 19:
        print('Building VGG-19 model...')
        x = vgg_2block(x,   3,  64, 'vgg_block1')
        x = vgg_2block(x,  64, 128, 'vgg_block2')
        x = vgg_4block(x, 128, 256, 'vgg_block3')
        x = vgg_4block(x, 256, 512, 'vgg_block4')
        x = vgg_4block(x, 512, 512, 'vgg_block5')
    
    else:
        assert False, 'VGG model should have 16 or 19 layers!'
    
    x = ad.array_reshape_op(x, (-1, 7 *7 * 512))
    x = vgg_fc(x,  7 * 7 * 512, 4096, 'vgg_fc1')
    x = vgg_fc(x, 4096, 4096, 'vgg_fc2')
    y = vgg_fc(x, 4096, 1000, 'vgg_fc3')

    loss = ad.softmaxcrossentropy_op(y, y_)

    return loss, y

def vgg(batch_size, num_layers, policy = "None"):
    global variable_list, val_list
    variable_list = []
    val_list = []

    X = ad.Variable(name='X')
    X_val = np.empty(shape=(batch_size, 3, 224, 224), dtype=np.float32)
    # X_val = ndarray.array(X_val, ctx=executor_ctx)
    y_ = ad.Variable(name='y_')
    y_val = np.empty(shape=(batch_size, 1000), dtype=np.float32)
    # y_val = ndarray.array(y_val, ctx=executor_ctx)

    loss, y = vgg_model(X, y_, num_layers)
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
    return (end - start) / 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-l','--layer', type=int, default=0)
    parser.add_argument('-p','--policy', default='None')
    args = parser.parse_args()

    policy = args.policy
    layer = args.layer

    # batch_size = 450
    # execution_time = vgg(batch_size, layer, policy = policy)
    output_file_name = "/home/xiaonan/microop/Athena/exp/" + "vgg" + str(layer) + "/" + policy + "_batchsize_with_time.txt"
    output_file = open(output_file_name, "a+", buffering=1)
    output_file.write("Policy: {}, on VGG{}\n".format(policy, layer))

    for batch_size in range(1, 1000, 1):
        execution_time = vgg(batch_size, layer, policy = policy)
        print("Batch size: {} , time: {} s\n".format(batch_size, execution_time))
        output_file.write("Batch size: {} , time: {} s\n".format(batch_size, execution_time))
    output_file.close()
