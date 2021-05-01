from athena import gpu_ops as ad
from athena import ndarray
import time
import numpy as np
import argparse

executor_ctx = ndarray.gpu(0)
variable_list = []
val_list = []

rand = np.random.RandomState(seed=123)

def get_variable(name, size, trainable = True):
    global variable_list, val_list
    x = ad.Variable(name=name, trainable = trainable)
    x_val = rand.normal(scale=0.1, size=size)
    x_val = ndarray.array(x_val, ctx=executor_ctx)
    variable_list.append(x)
    val_list.append(x_val)
    return x

def lstm_model(x, y_):
    '''
    LSTM model, for MNIST dataset.
    Parameters:
        x: Variable(athena.gpu_ops.Node.Node), shape (N, dims)
        y_: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    Return:
        loss: Variable(athena.gpu_ops.Node.Node), shape (1,)
        y: Variable(athena.gpu_ops.Node.Node), shape (N, num_classes)
    '''

    print("Building LSTM model...")
    diminput = 28
    dimhidden = 128
    dimoutput = 10
    nsteps = 28

    forget_gate_w = get_variable(name="lstm_forget_gate_w", size=(diminput, dimhidden))
    forget_gate_u = get_variable(name="lstm_forget_gate_u", size=(dimhidden, dimhidden))
    forget_gate_b = get_variable(name="lstm_forget_gate_b", size=(dimhidden,))
    input_gate_w = get_variable(name="lstm_input_gate_w", size=(diminput,dimhidden))
    input_gate_u = get_variable(name="lstm_input_gate_u", size=(dimhidden,dimhidden))
    input_gate_b = get_variable(name="lstm_input_gate_b", size=(dimhidden,))
    output_gate_w = get_variable(name="lstm_output_gate_w", size=(diminput, dimhidden))
    output_gate_u = get_variable(name="lstm_output_gate_u", size=(dimhidden, dimhidden))
    output_gate_b = get_variable(name="lstm_output_gate_b", size=(dimhidden,))
    tanh_w = get_variable(name="lstm_tanh_w", size=(diminput, dimhidden))
    tanh_u = get_variable(name="lstm_tanh_u", size=(dimhidden, dimhidden))
    tanh_b = get_variable(name="lstm_tanh_b", size=(dimhidden,))
    out_weights = get_variable(name="lstm_out_weight", size=(dimhidden, dimoutput))
    out_bias = get_variable(name="lstm_out_bias", size=(dimoutput,))
    initial_state = get_variable(name="initial_state", size=(1,), trainable = False)

    for i in range(nsteps):
        cur_x = ad.slice_op(x, (0, i * diminput), (-1, diminput))        
        # forget gate
        if i == 0:
            temp = ad.matmul_op(cur_x, forget_gate_w)
            last_c_state = ad.broadcasttoTF_op(initial_state, temp)
            last_h_state = ad.broadcasttoTF_op(initial_state, temp)
            cur_forget = ad.matmul_op(last_h_state, forget_gate_u) + temp
        else:
            cur_forget = ad.matmul_op(last_h_state, forget_gate_u) + ad.matmul_op(cur_x, forget_gate_w)
        cur_forget = cur_forget + ad.broadcasttoTF_op(forget_gate_b, cur_forget)
        cur_forget = ad.sigmoid_op(cur_forget)
        # input gate
        cur_input = ad.matmul_op(last_h_state, input_gate_u) + ad.matmul_op(cur_x, input_gate_w)
        cur_input = cur_input + ad.broadcasttoTF_op(input_gate_b, cur_input)
        cur_input = ad.sigmoid_op(cur_input)
        # output gate
        cur_output = ad.matmul_op(last_h_state, output_gate_u) + ad.matmul_op(cur_x, output_gate_w)
        cur_output = cur_output + ad.broadcasttoTF_op(output_gate_b, cur_output)
        cur_output = ad.sigmoid_op(cur_output)
        # tanh
        cur_tanh = ad.matmul_op(last_h_state, tanh_u) + ad.matmul_op(cur_x, tanh_w)
        cur_tanh = cur_tanh + ad.broadcasttoTF_op(tanh_b, cur_tanh)
        cur_tanh = ad.tanh_op(cur_tanh)

        last_c_state = ad.mul_op(last_c_state, cur_forget) + ad.mul_op(cur_input, cur_tanh)
        last_h_state = ad.tanh_op(last_c_state) * cur_output

    x = ad.matmul_op(last_h_state, out_weights)
    y = x + ad.broadcasttoTF_op(out_bias, x)
    loss = ad.softmaxcrossentropy_op(y, y_)
    return loss, y

def lstm(batch_size, policy = "None"):
    global variable_list, val_list
    variable_list = []
    val_list = []

    X = ad.Variable(name='X')
    X_val = np.empty(shape=(batch_size, 28*28), dtype=np.float32)
    # X_val = ndarray.array(X_val, ctx=executor_ctx)
    y_ = ad.Variable(name='y_')
    y_val = np.empty(shape=(batch_size, 10), dtype=np.float32)
    # y_val = ndarray.array(y_val, ctx=executor_ctx)

    loss, y = lstm_model(X, y_)
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
    
    for i in range(5):
        if i == 1:
            start = time.time()
        grad_val_list = executor.run(feed_dict)
    end = time.time()
    return (end - start) / 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-p','--policy', default='None')
    args = parser.parse_args()
    policy = args.policy
    # batch_size = 500
    # policy = args.policy
    # execution_time = lstm(batch_size, policy = policy)

    # output_file_name = "/home/xiaonan/microop/Athena/exp/" + "lstm" +  "/" + policy + "_batchsize_with_time.txt"
    # output_file = open(output_file_name, "w", buffering=1)
    # output_file.write("Policy: {}, on LSTM\n".format(policy))

    for batch_size in range(10, 1000, 1):
        execution_time = lstm(batch_size, policy = policy)
        print("Batch size: {} , time: {} s\n".format(batch_size, execution_time))
    #     output_file.write("Batch size: {} , time: {} s\n".format(batch_size, execution_time))
    # output_file.close()
