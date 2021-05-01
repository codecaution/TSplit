from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad
import numpy as np

def test_batch_size():
    print("test batch size")
    output_file = open("./relu_exp/relu_batch_size_2.txt", 'w', buffering=1)
    output_file.write("Input size = batch_size, 64, 224, 224\n")
    output_file.write("Output size = batch_size, 64, 224, 224\n")

    for batch_size in range(1, 1500):
        N, C, H, W = batch_size, 64, 224, 224
        X = ad.Variable(name = "X")
        reluOp = ad.relu_op(X)
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))

        output_val = ndarray.empty((N, C, H, W), ctx = ndarray.gpu(0))

        reluOp.op.profile(reluOp, [X_val], output_val, is_static = False)
        # print("Batch size = %d , execute time = %.5f ms\n"%(batch_size, reluOp.profiler.time))
        output_file.write("Batch size = %d , execute time = %.5f ms\n"%(batch_size, reluOp.profiler.time))

    output_file.close()

def test_channel():
    print("test channel")
    output_file = open("./relu_exp/relu_channel_size.txt", 'w', buffering=1)
    output_file.write("Input size = 128, channel_size, 32, 32\n")
    output_file.write("Output size = 128, channel_size, 32, 32\n")
    for channel_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4096*2]:
        N, C, H, W = 128, channel_size, 32, 32
        X = ad.Variable(name = "X")
        reluOp = ad.relu_op(X)
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))

        output_val = ndarray.empty((N, C, H, W), ctx = ndarray.gpu(0))

        for i in range(10):
            reluOp.op.profile(reluOp, [X_val], output_val, is_static = False)
        print("channel size = {}, profiler time = {} ms".format(channel_size, reluOp.profiler.time))
        output_file.write("%d %.5f\n"%(channel_size, reluOp.profiler.time))
    output_file.close()

def test_hw():
    print("test height and width")
    output_file = open("relu_height_width.txt", 'w')
    for height in [8, 16, 32, 64, 128, 256]:
        N, C, H, W = 128, 128, height, height
        X = ad.Variable(name = "X")
        reluOp = ad.relu_op(X)
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))

        output_val = ndarray.empty((N, C, H, W), ctx = ndarray.gpu(0))

        for i in range(10):
            reluOp.op.profile(reluOp, [X_val], output_val, is_static = False)
        print("height and width = {}, profiler time = {} ms".format(height, reluOp.profiler.time))
        output_file.write("%d %.5f\n"%(height, reluOp.profiler.time))
    output_file.close()

if __name__ == "__main__":
    test_batch_size()
    # test_channel()
    # test_hw()