import numpy as np
from . import ndarray
import gpu_links as gpu_op
from . import gpu_ops as ad
from .gpu_ops.Node import Op, NAME_RULE


class Optimizer(object):
    """Optimizers."""
    def __init__(self, ctx=None):
        self.learning_rate = 0
        self.ctx = ctx
        self.tensors = None
        self.params = None
        self.initiated = False

    def get_var_list(self, loss):
        from .gpu_ops.Variable import PlaceholderOp
        def topo_sort_dfs(node, visited, var_list):
            if node in visited:
                return
            visited.add(node)
            if isinstance(node.op, PlaceholderOp) and node.trainable:
                var_list.append(node)
                return
            for n in node.inputs:
                topo_sort_dfs(n, visited, var_list)

        visited = set()
        trainable_vars = []
        topo_sort_dfs(loss, visited, trainable_vars)
        return trainable_vars

    def initiate_states(self):
        raise NotImplementedError

    def initiate_vars(self, loss, var_list):
        assert not self.initiated, "Optimizer already initiated."
        if not var_list:
            var_list = self.get_var_list(loss)
        for tensor in var_list:
            assert tensor.tensor_value is not None, "Parameters %s not initialized." % tensor.name
            if ndarray.is_gpu_ctx(self.ctx):
                if isinstance(tensor.tensor_value, np.ndarray):
                    tensor.tensor_value = ndarray.array(tensor.tensor_value, ctx=self.ctx)
                else:
                    assert isinstance(tensor.tensor_value,
                                      ndarray.NDArray), "Parameter %s type should be ndarray.NDArray." % tensor.name
            else:
                if isinstance(tensor.tensor_value, ndarray.NDArray):
                    tensor.tensor_value = tensor.tensor_value.asnumpy()
                else:
                    assert isinstance(tensor.tensor_value,
                                      np.ndarray), "Parameter %s type should be numpy.ndarray." % tensor.name
        self.tensors = var_list
        self.params = [tensor.tensor_value for tensor in var_list]
        self.initiate_states()
        self.initiated = True

    def minimize(self, loss, var_list=None):
        self.initiate_vars(loss, var_list)
        grads = ad.gradients(loss, self.tensors)
        optimizer_node = OptimizerOp()(grads, self)
        return optimizer_node


class OptimizerOp(Op):
    def __call__(self, grads, optimizer):
        new_node = Op.__call__(self)
        new_node.inputs = grads
        new_node.name = "Optimizer_%s" % (optimizer.name)
        self.optimizer = optimizer
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True, stream_handle=None):
        assert output_val is None
        self.optimizer.update(input_vals, use_numpy, stream_handle)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return None


class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, ctx=None):
        super(SGDOptimizer, self).__init__(ctx)
        self.learning_rate = learning_rate
        self.name = 'SGD'

    def initiate_states(self):
        pass

    def update(self, grads, swap=False, stream_handle=None):
        """Helper GPU SGD update method. Avoids copying NDArray to cpu."""
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            # print grads[i].asnumpy()
            # print params[i].asnumpy()
            if ndarray.is_gpu_ctx(self.ctx):
                if swap:
                    self.params[i][:] = self.params[i] - self.learning_rate * grads[i]
                else:
                    assert isinstance(self.params[i], ndarray.NDArray)
                    assert isinstance(grads[i], ndarray.NDArray)
                    gpu_op.sgd_update(self.params[i], grads[i], self.learning_rate, stream_handle)
            else:
                from ._base import DNNL_LIB
                if DNNL_LIB['cpu_SGDOptimizerUpdate']:
                    from .cpu_links import sgd_update as cpu_sgd_update
                    from .ndarray import numpyasdlarrayhandle
                    param = numpyasdlarrayhandle(self.params[i])
                    grad = numpyasdlarrayhandle(grads[i])
                    cpu_sgd_update(param, grad, self.learning_rate)
                else:
                    self.params[i][:] = self.params[i] - self.learning_rate * grads[i]


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9, nesterov=False, ctx=None):
        super(MomentumOptimizer, self).__init__(ctx)
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.name = "Momentum"

    def initiate_states(self):
        if ndarray.is_gpu_ctx(self.ctx):
            self.velocity = [ndarray.array(np.zeros(p.shape, dtype=np.float32), self.ctx) for p in self.params]
        else:
            self.velocity = [np.zeros(p.shape) for p in self.params]

    def update(self, grads, swap=False, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            # print grads[i].asnumpy()
            # print params[i].asnumpy()
            if ndarray.is_gpu_ctx(self.ctx) and not swap:
                assert isinstance(self.params[i], ndarray.NDArray)
                assert isinstance(grads[i], ndarray.NDArray)
                assert isinstance(self.velocity[i], ndarray.NDArray)
                gpu_op.momentum_update(self.params[i], grads[i], self.velocity[i], self.learning_rate, self.momentum,
                                       self.nesterov, stream_handle)

            elif swap:
                if self.nesterov:
                    lr_grads = -self.learning_rate * grads[i]
                    self.velocity[i][:] = self.momentum * (self.velocity[i] + lr_grads)
                    self.params[i][:] = self.params[i] + self.velocity[i] + lr_grads
                else:
                    self.velocity[i][:] = self.momentum * self.velocity[i] - self.learning_rate * grads[i]
                    self.params[i][:] = self.params[i] + self.velocity[i]

            else:
                from ._base import DNNL_LIB
                if DNNL_LIB['cpu_MomentumOptimizerUpdate']:
                    from .cpu_links import momentum_update as cpu_momentum_update
                    from .ndarray import numpyasdlarrayhandle
                    param = numpyasdlarrayhandle(self.params[i])
                    grad = numpyasdlarrayhandle(grads[i])
                    velocity=numpyasdlarrayhandle(self.velocity[i])
                    cpu_momentum_update(param, grad, velocity,self.learning_rate,self.momentum,
                                       self.nesterov)
                else:
                    if self.nesterov:
                        lr_grads = -self.learning_rate * grads[i]
                        self.velocity[i][:] = self.momentum * (self.velocity[i] + lr_grads)
                        self.params[i][:] = self.params[i] + self.velocity[i] + lr_grads
                    else:
                        self.velocity[i][:] = self.momentum * self.velocity[i] - self.learning_rate * grads[i]
                        self.params[i][:] = self.params[i] + self.velocity[i]


class AdaGradOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, initial_accumulator_value=0.0, eps=1e-7, ctx=None):
        assert learning_rate >= 0, \
            "learning rate must be non-negative"
        assert initial_accumulator_value >= 0.0, \
            "initial accumulator value must be non-negative"
        assert eps > 0.0, \
            "epsilon must be positive"
        super(AdaGradOptimizer, self).__init__(ctx)
        self.learning_rate = learning_rate
        self.initial_accumulator_value = initial_accumulator_value
        self.eps = eps
        self.name = "AdaGrad"

    def initiate_states(self):
        if ndarray.is_gpu_ctx(self.ctx):
            self.accumulator_value = [ndarray.array(np.full(p.shape, self.initial_accumulator_value), self.ctx) for p in
                                      self.params]
        else:
            self.accumulator_value = [np.full(p.shape, self.initial_accumulator_value) for p in self.params]

    def update(self, grads, swap=False, stream_handle=None):
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        for i in range(params_size):
            if ndarray.is_gpu_ctx(self.ctx) and not swap:
                assert isinstance(self.params[i], ndarray.NDArray)
                assert isinstance(grads[i], ndarray.NDArray)
                gpu_op.adagrad_update(self.params[i], grads[i], self.accumulator_value[i], self.learning_rate, self.eps,
                                      stream_handle)
            elif swap:
                local_acc = self.accumulator_value[i] + np.power(grads[i], 2)
                self.accumulator_value[i][:] = local_acc
                self.params[i][:] = \
                    self.params[i] - self.learning_rate * grads[i] / (np.sqrt(local_acc) + self.eps)
            else:
                from ._base import DNNL_LIB
                if DNNL_LIB['cpu_AdaGradOptimizerUpdate']:
                    from .cpu_links import adagrad_update as cpu_adagrad_update
                    from .ndarray import numpyasdlarrayhandle
                    param = numpyasdlarrayhandle(self.params[i])
                    grad = numpyasdlarrayhandle(grads[i])
                    accumulation=numpyasdlarrayhandle(self.accumulator_value[i])
                    cpu_adagrad_update(param, grad, accumulation,self.learning_rate,self.eps)
                else:
                    self.accumulator_value[i][:] = self.accumulator_value[i] + np.power(grads[i], 2)
                    self.params[i][:] = \
                        self.params[i] - self.learning_rate * grads[i] / (np.sqrt(self.accumulator_value[i]) + self.eps)


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7, ctx=None):
        super(AdamOptimizer, self).__init__(ctx)
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta1_t = 1.0
        self.beta2 = beta2
        self.beta2_t = 1.0
        self.epsilon = epsilon
        self.name = "Adam"

    def initiate_states(self):
        if ndarray.is_gpu_ctx(self.ctx):
            self.m = [ndarray.array(np.zeros(p.shape), self.ctx) for p in self.params]
            self.v = [ndarray.array(np.zeros(p.shape), self.ctx) for p in self.params]
        else:
            self.m = [np.zeros(p.shape) for p in self.params]
            self.v = [np.zeros(p.shape) for p in self.params]

    def update(self, grads, swap=False, stream_handle=None):
        """Helper GPU SGD update method. Avoids copying NDArray to cpu."""
        assert self.initiated is True
        params_size = len(self.params)
        assert params_size == len(grads)
        self.beta1_t *= self.beta1
        self.beta2_t *= self.beta2
        for i in range(params_size):
            # print grads[i].asnumpy()
            # print params[i].asnumpy()
            if ndarray.is_gpu_ctx(self.ctx) and not swap:
                assert isinstance(self.params[i], ndarray.NDArray)
                assert isinstance(grads[i], ndarray.NDArray)
                assert isinstance(self.m[i], ndarray.NDArray)
                assert isinstance(self.v[i], ndarray.NDArray)
                gpu_op.adam_update(self.params[i], grads[i], self.m[i], self.v[i], self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon, stream_handle)
            elif swap:
                local_m = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
                local_v = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] * grads[i]
                self.m[i][:] = local_m
                self.v[i][:] = local_v
                mc = local_m / (1 - self.beta1_t)
                vc = local_v / (1 - self.beta2_t)
                self.params[i][:] = self.params[i] - self.learning_rate * mc / (np.sqrt(vc) + self.epsilon)
            else:
                from ._base import DNNL_LIB
                if DNNL_LIB['cpu_AdaGradOptimizerUpdate']:
                    from .cpu_links import adam_update as cpu_adam_update
                    from .ndarray import numpyasdlarrayhandle
                    param = numpyasdlarrayhandle(self.params[i])
                    grad = numpyasdlarrayhandle(grads[i])
                    expavg = numpyasdlarrayhandle(self.m[i])
                    expavgsq = numpyasdlarrayhandle(self.v[i])
                    cpu_adam_update(param, grad, expavg,expavgsq, self.learning_rate, self.beta1,
                                   self.beta2, self.beta1_t, self.beta2_t, self.epsilon)
                else:
                    self.m[i][:] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
                    self.v[i][:] = self.beta2 * self.v[i] + (1 - self.beta2) * grads[i] * grads[i]
                    mc = self.m[i] / (1 - self.beta1_t)
                    vc = self.v[i] / (1 - self.beta2_t)
                    self.params[i][:] = self.params[i] - self.learning_rate * mc / (np.sqrt(vc) + self.epsilon)