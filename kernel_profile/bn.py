from athena import ndarray
from athena import gpu_links as gpu_op
from athena import gpu_ops as ad
import numpy as np

def init():
    # init cudnn handle
        N, C, H, W = 100, 128, 32, 32
        X = ad.Variable(name = "X")
        bn_scale= ad.Variable(name = "bn_scale")
        bn_bias = ad.Variable(name = "bn_bias")
        bn_op = ad.batch_normalization_op(X, bn_scale, bn_bias)
        
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        bn_scale_val = np.random.normal(scale=0.1, size=(1, 128, 1, 1))
        bn_bias_val = np.random.normal(scale=0.1, size=(1, 128, 1, 1))

        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))
        bn_scale_val = ndarray.array(bn_scale_val, ctx = ndarray.gpu(0))
        bn_bias_val = ndarray.array(bn_bias_val, ctx = ndarray.gpu(0))

        output_val = ndarray.empty((N, C, H, W), ctx = ndarray.gpu(0))

        bn_op.op.profile(bn_op, [X_val, bn_scale_val, bn_bias_val], output_val, is_static = False)
        
def test_batch_size():
    print("test batch size")
    output_file = open("./bn_exp/bn_batch_size_2.txt", 'w', buffering=1)
    output_file.write("Input size = batch_size, 64, 224, 224\n")
    output_file.write("Output size = batch_size, 64, 224, 224\n")

    for batch_size in range(1, 1500):
        N, C, H, W = batch_size, 64, 224, 224
        X = ad.Variable(name = "X")
        bn_scale= ad.Variable(name = "bn_scale")
        bn_bias = ad.Variable(name = "bn_bias")
        bn_op = ad.batch_normalization_op(X, bn_scale, bn_bias)
        
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        bn_scale_val = np.random.normal(scale=0.1, size=(1, C, 1, 1))
        bn_bias_val = np.random.normal(scale=0.1, size=(1, C, 1, 1))

        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))
        bn_scale_val = ndarray.array(bn_scale_val, ctx = ndarray.gpu(0))
        bn_bias_val = ndarray.array(bn_bias_val, ctx = ndarray.gpu(0))
        output_val = ndarray.empty((N, C, H, W), ctx = ndarray.gpu(0))

        bn_op.op.profile(bn_op, [X_val, bn_scale_val, bn_bias_val], output_val, is_static = False)
        output_file.write("Batch size = %d , execute time = %.5f ms\n"%(batch_size, bn_op.profiler.time))
    output_file.close()

def test_channel():
    print("test channel")
    output_file = open("bn_channel_size.txt", 'w')
    for channel_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 4096*2]:
        N, C, H, W = 128, channel_size, 32, 32
        X = ad.Variable(name = "X")
        bn_scale= ad.Variable(name = "bn_scale")
        bn_bias = ad.Variable(name = "bn_bias")
        bn_op = ad.batch_normalization_op(X, bn_scale, bn_bias)
        
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        bn_scale_val = np.random.normal(scale=0.1, size=(1, C, 1, 1))
        bn_bias_val = np.random.normal(scale=0.1, size=(1, C, 1, 1))

        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))
        bn_scale_val = ndarray.array(bn_scale_val, ctx = ndarray.gpu(0))
        bn_bias_val = ndarray.array(bn_bias_val, ctx = ndarray.gpu(0))

        output_val = ndarray.empty((N, C, H, W), ctx = ndarray.gpu(0))

        for i in range(10):
            bn_op.op.profile(bn_op, [X_val, bn_scale_val, bn_bias_val], output_val, is_static = False)
        print("channel size = {}, profiler time = {} ms".format(channel_size, bn_op.profiler.time))
        output_file.write("%d %.5f\n"%(channel_size, bn_op.profiler.time))
    output_file.close()

def test_hw():
    print("test height and width")
    output_file = open("bn_height_width.txt", 'w')
    for height in [8, 16, 32, 64, 128, 256]:
        N, C, H, W = 128, 128, height, height
        X = ad.Variable(name = "X")
        bn_scale= ad.Variable(name = "bn_scale")
        bn_bias = ad.Variable(name = "bn_bias")
        bn_op = ad.batch_normalization_op(X, bn_scale, bn_bias)
        
        rand = np.random.RandomState(seed=123)
        X_val = np.random.normal(scale=0.1, size=(N, C, H, W))
        bn_scale_val = np.random.normal(scale=0.1, size=(1, C, 1, 1))
        bn_bias_val = np.random.normal(scale=0.1, size=(1, C, 1, 1))

        X_val = ndarray.array(X_val, ctx = ndarray.gpu(0))
        bn_scale_val = ndarray.array(bn_scale_val, ctx = ndarray.gpu(0))
        bn_bias_val = ndarray.array(bn_bias_val, ctx = ndarray.gpu(0))

        output_val = ndarray.empty((N, C, H, W), ctx = ndarray.gpu(0))

        for i in range(10):
            bn_op.op.profile(bn_op, [X_val, bn_scale_val, bn_bias_val], output_val, is_static = False)
        print("height and weight size = {}, profiler time = {} ms".format(height, bn_op.profiler.time))
        output_file.write("%d %.5f\n"%(height, bn_op.profiler.time))
    output_file.close()

if __name__ == "__main__":
    init()
    test_batch_size()
    # test_channel()
    # test_hw()