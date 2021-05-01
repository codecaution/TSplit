from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad
import numpy as np


def init():
    # init with cudnn handle
    N, C, H, W = 100, 64, 32, 32
    f_N, f_C, f_H, f_W = 128, C, 3, 3
    out_N = N
    out_C = f_N
    out_H = (H + 2 * 1 - f_H) + 1
    out_W = (W + 2 * 1 - f_W) + 1
    W0 = ad.Variable(name = "W0")
    X = ad.Variable(name = "X")
    Y = ad.Variable(name = "Y")

    conv_op = ad.conv2d_op(X, W0, padding = 1, stride = 1)
    rand = np.random.RandomState(seed=123)
    X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
    W0_val = np.random.normal(scale=0.1, size=(f_N, f_C, f_H, f_W))

    X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))
    W0_val = ndarray.array(W0_val, ctx = ndarray.gpu(0))
    output_val = ndarray.empty((out_N, out_C, out_H, out_W), ctx = ndarray.gpu(0))
    
    conv_op.op.profile(conv_op, [X_val, W0_val], output_val, is_static = False) 
     
def test_batch_size():
    print("test batch size")
    output_file = open("./conv_exp/convForward_batch_size_2.txt", 'w', buffering=1)
    output_file.write("Input size = batch_size, 64, 224, 224\n")
    output_file.write("kernel size = 128, 64, 3, 3\n")
    output_file.write("Output size = 128, 128, 224, 224\n")

    for batch_size in range(1, 1500):
        N, C, H, W = batch_size, 64, 32, 32
        f_N, f_C, f_H, f_W = 128, C, 3, 3
        out_N = N
        out_C = f_N
        out_H = (H + 2 * 1 - f_H) + 1
        out_W = (W + 2 * 1 - f_W) + 1
        W0 = ad.Variable(name = "W0")
        X = ad.Variable(name = "X")
        Y = ad.Variable(name = "Y")

        conv_op = ad.conv2d_op(X, W0, padding = 1, stride = 1)
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        W0_val = np.random.normal(scale=0.1, size=(f_N, f_C, f_H, f_W))

        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))
        W0_val = ndarray.array(W0_val, ctx = ndarray.gpu(0))
        output_val = ndarray.empty((out_N, out_C, out_H, out_W), ctx = ndarray.gpu(0))
        
        conv_op.op.profile(conv_op, [X_val, W0_val], output_val, is_static = False)
        # conv_op.op.profile(conv_op, [X_val, W0_val], output_val, is_static = False)
        # print("Conv: batch size = {}, profiler time = {} ms".format(batch_size, conv_op.profiler.time))
        
        output_file.write("Batch size = %d , execute time = %.5f ms\n"%(batch_size, conv_op.profiler.time))
        
        """
        conv_grad_data_op = ad.conv2d_gradient_of_data_op(W0, Y, padding = 1, stride = 1)
        
        for i in range(10):
            conv_grad_data_op.op.profile(conv_grad_data_op, [W0_val, output_val], X_val, is_static = False)
        print("Conv gradient of data: batch size = {}, profiler time = {} ms".format(batch_size, conv_grad_data_op.profiler.time))
        
        conv_grad_filter_op = ad.conv2d_gradient_of_filter_op(X, Y, padding = 1, stride = 1)

        for i in range(10):
            conv_grad_filter_op.op.profile(conv_grad_filter_op, [X_val, output_val], W0_val, is_static = False)

        print("Conv gradient of filter: batch size = {}, profiler time = {} ms".format(batch_size, conv_grad_filter_op.profiler.time))

        print("")        
        """        
        # output_file.write("%d %.5f\n"%(batch_size, conv_op.profiler.time))
    output_file.close()

def test_channel():
    print("test channel")
    output_file = open("conv_channel_size.txt", 'w')
    for channel_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4096*2]:
        N, C, H, W = 128, channel_size, 32, 32
        f_N, f_C, f_H, f_W = 64, C, 3, 3
        out_N = N
        out_C = f_N
        out_H = (H + 2 * 1 - f_H) + 1
        out_W = (W + 2 * 1 - f_W) + 1
        W0 = ad.Variable(name = "W0")
        X = ad.Variable(name = "X")
        conv_op = ad.conv2d_op(X, W0, padding = 1, stride = 1)
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        W0_val = np.random.normal(scale=0.1, size=(f_N, f_C, f_H, f_W))

        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))
        W0_val = ndarray.array(W0_val, ctx = ndarray.gpu(0))
        output_val = ndarray.empty((out_N, out_C, out_H, out_W), ctx = ndarray.gpu(0))
        # print(out_N, out_C, out_H, out_W)
        # print(dir(conv_op))
        for i in range(10):
            conv_op.op.profile(conv_op, [X_val, W0_val], output_val, is_static = False)
        print("profiler time = {} ms".format(conv_op.profiler.time))
        output_file.write("%d %.5f\n"%(channel_size, conv_op.profiler.time))
    output_file.close()

def test_hw():
    print("test height and width")
    output_file = open("conv_height_width.txt", 'w')
    for height in [8, 16, 32, 64, 128, 256]:
        N, C, H, W = 128, 128, height, height
        f_N, f_C, f_H, f_W = 64, C, 3, 3
        out_N = N
        out_C = f_N
        out_H = (H + 2 * 1 - f_H) + 1
        out_W = (W + 2 * 1 - f_W) + 1
        W0 = ad.Variable(name = "W0")
        X = ad.Variable(name = "X")
        conv_op = ad.conv2d_op(X, W0, padding = 1, stride = 1)
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        W0_val = np.random.normal(scale=0.1, size=(f_N, f_C, f_H, f_W))

        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))
        W0_val = ndarray.array(W0_val, ctx = ndarray.gpu(0))
        output_val = ndarray.empty((out_N, out_C, out_H, out_W), ctx = ndarray.gpu(0))
        # print(out_N, out_C, out_H, out_W)
        # print(dir(conv_op))
        for i in range(10):
            conv_op.op.profile(conv_op, [X_val, W0_val], output_val, is_static = False)
        print("HW = {}, profiler time = {} ms".format(height, conv_op.profiler.time))
        output_file.write("%d %.5f\n"%(height, conv_op.profiler.time))
    output_file.close()

if __name__ == "__main__":
    init()
    test_batch_size()
    # test_channel()
    # test_hw()
