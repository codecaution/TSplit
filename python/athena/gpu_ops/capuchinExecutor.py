""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
import numpy as np
from .Node import Op
from .. import ndarray
from ..stream import *
from ..logger import Logger
from .._base import get_array_memory
from .. import profiler

import ctypes
import os
from pynvml import *
FLAG_SHOW_GRAPH = False
G_NODE_ID = 0
NAME_RULE = 1

def communicate_init(worker_num, worker_id, source_ip, target_ip):
    global lib_communicate
    # lib_communicate.DL_Connect_Init(2, 0, "*:4001", "localhost:4002")
    # lib_communicate.DL_Connect_Init(2, 1, "*:4002", "localhost:4001")
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "lib_communication.so")
    lib_communicate = ctypes.cdll.LoadLibrary(path_to_so_file)
    lib_communicate.DL_Connect_Init(
        worker_num, worker_id, source_ip, target_ip)

def communicate_finish():
    lib_communicate.DL_Communicate_Close()

class Distributed_CommunicateOp(Op):
    def __call__(self, nodeA):
        new_node = Op.__call__(self)
        new_node.inputs = [nodeA]
        new_node.name = "Distributed_Communicate(%s)" % (nodeA.name)
        # print nodeA.name
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        after_reduce_gradient_cpu = ndarray.empty(
            shape=output_val.shape, ctx=ndarray.cpu(0))
        if use_numpy:
            gradient_val_cpu = ndarray.array(input_vals[0], ctx=ndarray.cpu(0))
        else:
            gradient_val_cpu = ndarray.array(
                input_vals[0].asnumpy(), ctx=ndarray.cpu(0))
        # print gradient_val_cpu.asnumpy()
        lib_communicate.DL_Communicate_Init(gradient_val_cpu.handle)
        lib_communicate.DL_Communicate(
            gradient_val_cpu.handle, after_reduce_gradient_cpu.handle)
        # print after_reduce_gradient_cpu.asnumpy()
        if use_numpy:
            output_val[:] = after_reduce_gradient_cpu.asnumpy()
        else:
            after_reduce_gradient_cpu.copyto(output_val)

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]


distributed_communicate_op = Distributed_CommunicateOp()

class capuchinExecutor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_list, ctx = None, streams = None, policy = None):
        """
        Parameters
        ----------
        eval_node_list: list of nodes whose values need to be computed.
        ctx: runtime DLContext, default is None which means np.ndarray on cpu
        topo_order: list of nodes in topological order
        node_to_shape_map: dict from node to shape of the node
        node_to_arr_map: dict from node to ndarray.NDArray allocated for node
        feed_shapes: shapes of feed_dict from last run(...)
        """
        self.eval_node_list = eval_node_list
        self.ctx = ctx
        if streams is None:
            self.streams = []
            # h2d stream
            self.streams.append(create_stream_handle(ctx))
            # computation stream
            self.streams.append(create_stream_handle(ctx))
            # d2h stream
            self.streams.append(create_stream_handle(ctx))
        else:
            assert (len(streams) == 3)
            self.streams = streams
        for stream in self.streams:
            stream.sync()
        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.node_to_cpu_arr_map = None
        self.feed_shapes = None
        self.node_to_event_map = None
        self.policy = policy
        if self.policy == 'swap':
            self.swap_queue = []

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""
        self.node_to_shape_map = {}

        for node in feed_shapes:
            self.node_to_shape_map[node] = feed_shapes[node]
        for node in self.topo_order:
            if node not in feed_shapes:
                # print(node.name)
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                self.node_to_shape_map[node] = node.op.infer_shape(
                    node, input_shapes)

    def capuchin_policy(self):
        assert (self.ctx is not None)
        # the interval of nodes which are kept in the GPU
        gpu_id = self.ctx.device_id
        nvmlInit()
        handle = nvmlDeviceGetHandleByIndex(gpu_id)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_free_memory = info.free
        nvmlShutdown()
        # print("gpu_free_memory", gpu_free_memory)
        
        loss_node = self.eval_node_list[0]
        self.loss_index = self.topo_order.index(loss_node)
        self.node_in_gpu = {}
        for node in self.topo_order:
            self.node_in_gpu[node] = False

        from .Variable import PlaceholderOp  # add for optimizer        
        for node in reversed(self.topo_order[:self.loss_index + 1]):
            # used_memory = node.used_memory
            # print(self.node_to_shape_map[node])
            used_memory = get_array_memory(self.node_to_shape_map[node])
            # print(used_memory)
            if isinstance(node.op, PlaceholderOp):
                gpu_free_memory -= used_memory
                self.node_in_gpu[node] = True

        for node in reversed(self.topo_order[:self.loss_index + 1]):
            if self.node_in_gpu[node] is True:
                continue
            # used_memory = node.used_memory
            # print(self.node_to_shape_map[node])
            used_memory = get_array_memory(self.node_to_shape_map[node])
            # print(used_memory)
            if gpu_free_memory > used_memory:
                # print(node, gpu_free_memory, used_memory)
                gpu_free_memory -= used_memory
                self.node_in_gpu[node] = True
            else:
                break

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.

        Implementation note:
        Option 1: Alloc a ndarray.NDArray per node that persists across run()
        Option 2: Implement a memory pool to reuse memory for nodes of same
                shapes. More details see Lecture 7.

        For both options, self.node_to_arr_map stores node->NDArray mapping to
        allow mapping to persist across multiple executor.run().

        Hint: use ndarray.empty(shape, ctx=self.ctx) to allocate NDArray.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        """TODO: Your code here"""

        for node, shape in self.node_to_shape_map.items():
            # add for optimizer, PlaceholderOp with values directly handled
            if isinstance(node.op, PlaceholderOp) and node.tensor_value is not None:
                arr = node.tensor_value
                if isinstance(node, np.ndarray) or (isinstance(node, ndarray.NDArray) and not ndarray.is_gpu_ctx(node.ctx)):
                    arr = ndarray.array(arr, ctx=self.ctx)
                    node.tensor_value = arr
                self.node_to_arr_map[node] = arr
            else:
                # add for optimizer, PlaceholderOp with values have None shape
                if shape is None:
                    self.node_to_arr_map[node] = None
                    continue
                if self.policy == 'vdnn' and self.node_in_gpu[node] is False:
                    print("shape = ", shape)
                    self.node_to_arr_map[node] = None
                    self.node_to_cpu_arr_map[node] = ndarray.empty(shape)
                else:
                    self.node_to_arr_map[node] = ndarray.empty(shape, ctx=self.ctx)

    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array

        Returns
        -------
        A list of values for nodes in eval_node_list. NDArray or np.ndarray.
        """
        def are_feed_shapes_equal(sa, sb):
            if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
                return False
            unmatched_item = set(sa.items()) ^ set(sb.items())
            return len(unmatched_item) == 0

        def is_forward_phase(node):
            return self.topo_order.index(node) <= self.loss_index
            
        # Assume self.ctx is None implies numpy array and numpy ops.

        use_numpy = self.ctx is None
    
        feed_shapes = {}
        for node, value in feed_dict.items():
            feed_shapes[node] = value.shape

        # if not use_numpy and self.node_to_event_map is None:
        #     self.node_to_event_map = {}
        #     for node, value in feed_dict.items():
        #         if isinstance(value, ndarray.NDArray) and not ndarray.is_gpu_ctx(value.ctx):
        #             self.node_to_event_map[node] = create_event_handle(self.ctx)
        
        #compute the output degree of each node
        self.output_degree = {}
        for node in self.topo_order:
            self.output_degree[node] = 0
            if node in feed_dict:
                continue
            for input_node in node.inputs:
                self.output_degree[input_node] += 1

        if(not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.node_to_arr_map = {}
            self.node_to_cpu_arr_map = {}
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            if (not use_numpy):
                # vdnn policy
                self.vdnn_policy()
                # 
                self.memory_plan(self.feed_shapes)

                self.node_to_event_map = {}
                # print("event node:")
                for node, value in feed_dict.items():
                    if isinstance(value, np.ndarray) or isinstance(value, ndarray.NDArray) and not ndarray.is_gpu_ctx(value.ctx):
                        self.node_to_event_map[node] = create_event_handle(self.ctx)
                        # print(node)
                for node_index, node in enumerate(self.topo_order):
                    if node_index <= self.loss_index and self.node_in_gpu[node] is False:
                        self.node_to_event_map[node] = create_event_handle(self.ctx)
                        # print(node)

        # for node in self.topo_order:
        #     print("output degree", node.name, self.output_degree[node])
        for node in self.topo_order:
            print(node.name, self.node_in_gpu[node])
        assert 1 == 2
        # print("use event:")
        for node, value in feed_dict.items():
            if use_numpy:
                # all values passed in feed_dict must be np.ndarray
                assert isinstance(value, np.ndarray)
                self.node_to_arr_map[node] = value
            else:
                # convert values to ndarray.NDArray if necessary
                if isinstance(value, np.ndarray) or (isinstance(value, ndarray.NDArray) and not ndarray.is_gpu_ctx(value.ctx)):
                    # print(node)
                    # if isinstance(value, np.ndarray):
                    #     print("np array")
                    # elif isinstance(value, ndarray.NDArray):
                    #     print("ndarray")
                    if isinstance(value, np.ndarray):
                        value = ndarray.array(value, ctx = ndarray.cpu(0))
                    self.node_to_arr_map[node].async_h2d(value, self.streams[0], self.node_to_event_map[node])
                elif isinstance(value, ndarray.NDArray):
                    self.node_to_arr_map[node] = value
                else:
                    assert False, "feed_dict value type not supported"
        
        loss_index = self.topo_order.index(self.eval_node_list[0])
        prefetch_k = 3
        # print("run")
        # TODO + swapin and timestamp + input should be cpu and gpu
        for node_index, node in enumerate(self.topo_order):
            # print(node_index, type(node))
            if node in feed_dict:
                continue
            # do prefetch process of backward node
            if node_index + prefetch_k > loss_index and node_index + prefetch_k < len(self.topo_order):
                target_node = self.topo_order[node_index + prefetch_k]
                for input_node in target_node.inputs:
                    if self.node_in_gpu[input_node] is False and is_forward_phase(input_node):
                        # print(input_node)
                        node_shape = self.node_to_cpu_arr_map[input_node].shape
                        value = self.node_to_cpu_arr_map[input_node]
                        self.node_to_arr_map[input_node] = ndarray.empty(node_shape, ctx = self.ctx)
                        self.node_to_arr_map[input_node].async_h2d(value, self.streams[0], self.node_to_event_map[input_node])

            input_vals = [self.node_to_arr_map[n] for n in node.inputs]
            input_events = [self.node_to_event_map[n] for n in node.inputs if n in self.node_to_event_map]
            for event in input_events:
                event.sync()
            # compute stream
            if use_numpy:
                node_val = np.empty(shape=self.node_to_shape_map[node])
            else:
                # forward
                if node_index <= loss_index:
                    node_val = self.node_to_arr_map[node]
                # backward
                else:
                    node_val = ndarray.empty(self.node_to_shape_map[node], ctx=self.ctx)
            # print(type(node_val))
            # print(node.name)
            # print(self.node_in_gpu[node])
            # print(self.topo_order.index(node))
            node.op.compute(node, input_vals, node_val, use_numpy, self.streams[1])
            self.node_to_arr_map[node] = node_val

            # if not used ever in this iteration, delete it
            # forward pass
            
            for input_node in node.inputs:
                self.output_degree[input_node] -= 1
                if node_index <= loss_index:
                    if self.output_degree[input_node] == 0 and self.node_in_gpu[input_node] is False:
                        # print("here")
                        node_value = self.node_to_arr_map[input_node]
                        node_event = self.node_to_event_map[input_node]
                        self.node_to_cpu_arr_map[input_node].async_d2h(node_value, self.streams[2], node_event)
                        del self.node_to_arr_map[input_node]
                else:
                    if not input_node in self.eval_node_list and self.output_degree[input_node] == 0 and self.node_in_gpu[input_node] is False:
                        # print(self.topo_order.index(input_node))
                        node_value = self.node_to_arr_map[input_node]
                        self.node_to_arr_map[input_node] = None
                        del node_value
        # print("epoch down")
        for stream in self.streams:
            stream.sync()
        if not use_numpy and convert_to_numpy_ret_vals:
            return [self.node_to_arr_map[n].asnumpy() for n in self.eval_node_list]
        return [self.node_to_arr_map[n] for n in self.eval_node_list]


def gradients(output_node, node_list, scheduler_policy=None):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    from . import OnesLike
    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [
        OnesLike.oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))

    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        #print len(node.name)
        #print len(node.inputs)
        #raw_input("\n\nPress the enter key to exit.")
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            # print node.name
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])

    grad_node_list = [node_to_output_grad[node] for node in node_list]
    return grad_node_list

'''
def distributed_gradients(output_node, node_list, scheduler_policy=None):
    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
    from .OnesLike import oneslike_op

    node_to_output_grads_list = {}
    node_to_output_grads_list[output_node] = [oneslike_op(output_node)]
    node_to_output_grad = {}
    # Traverse forward graph in reverse topological order
    reverse_topo_order = reversed(find_topo_sort([output_node]))
    for node in reverse_topo_order:
        output_grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = output_grad
        input_grads_list = node.op.gradient(node, output_grad)
        #print len(node.name)
        #print len(node.inputs)
        #raw_input("\n\nPress the enter key to exit.")
        for i in range(len(node.inputs)):
            if node.inputs[i] not in node_to_output_grads_list:
                node_to_output_grads_list[node.inputs[i]] = []
            # Calculate partial adjoint for input nodes.
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])
    if scheduler_policy == 'swap':
        for node in node_list:
            if node.swap:
                node_to_output_grad[node].swap = True
    # grad_node_list = [node_to_output_grad[node] for node in node_list]
    grad_node_list = [distributed_communicate_op(
        node_to_output_grad[node]) for node in node_list]
    return grad_node_list
'''
##################
# Helper Methods #
##################


def find_topo_sort(node_list):
    """Given a list of nodes, return a topo ordering of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a
    topological sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)


def broadcast_rule(shape_a, shape_b):
    """Return output shape of broadcast shape_a, shape_b.
    e.g. broadcast_rule((3,2), (4,3,2))
    returns output_shape = (4,3,2)

    Check out explanations and more examples at
    https://docs.scipy.org/doc/numpy-1.10.0/user/basics.broadcasting.html
    http://eli.thegreenplace.net/2015/broadcasting-arrays-in-numpy/
    """
    assert(isinstance(shape_a, tuple))
    assert(isinstance(shape_b, tuple))
    if len(shape_a) > len(shape_b):
        longer_shape, shorter_shape = shape_a, shape_b
    else:
        longer_shape, shorter_shape = shape_b, shape_a
    len_diff = len(longer_shape) - len(shorter_shape)
    for i in range(len_diff):
        # pad with leading 1s
        shorter_shape = (1,) + shorter_shape
    assert len(shorter_shape) == len(longer_shape)
    output_shape = list(longer_shape)
    for i in range(len(output_shape)):
        assert (shorter_shape[i] == longer_shape[i]) \
            or (shorter_shape[i] == 1) \
            or (longer_shape[i] == 1)
        output_shape[i] = max(shorter_shape[i], longer_shape[i])
    return tuple(output_shape)
