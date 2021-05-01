""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
import numpy as np
import scipy.sparse
from .Node import Op
from .. import ndarray
from .. import gpu_links as gpu_op
from ..simulator import Simulator
import ctypes
from .Conv2d import Conv2dOp, Conv2d_Gradient_of_DataOp, Conv2d_Gradient_of_FilterOp
import os
from pynvml import *
FLAG_SHOW_GRAPH = False
G_NODE_ID = 0
NAME_RULE = 1
from .Node import PROFILING_MODE
from .. import profiler

class profileExecutor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_list, ctx=None, policy=None):
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
        from .._base import _LIB
        _LIB.cudnn_init(ctypes.c_int(self.ctx.device_id), None)
        _LIB.cublas_init(ctypes.c_int(self.ctx.device_id), None)
        self.topo_order = find_topo_sort2(self.eval_node_list)
        rewrite_topoorder(self.topo_order)

        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.feed_shapes = None
        self.first_time = True
        self.policy = policy

    def infer_shape(self, feed_shapes):
        """Given shapes of feed_dict nodes, infer shape for all nodes in graph.

        Implementation note:
        Iteratively calls node.op.infer_shape to infer shapes.
        Node shapes stored in self.node_to_shape_map.

        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        self.node_to_shape_map = {}

        for node in self.topo_order:
            if node in feed_shapes:
                self.node_to_shape_map[node] = feed_shapes[node]
            else:
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                self.node_to_shape_map[node] = node.op.infer_shape(
                    node, input_shapes)

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
        assert (self.ctx is not None)
        self.node_to_arr_map = {}
        for node, shape in self.node_to_shape_map.items():
            self.node_to_arr_map[node] = ndarray.empty(shape, ctx=self.ctx)
    
    def profile(self, feed_dict = {}, is_print = False):
        """
        feed_dict: a dictionary of node->np.ndarray supplied by user.

        """
        node_to_shape_map = {}
        for node, value in sorted(feed_dict.items(),key=lambda val:val[0]):
            # convert values to ndarray.NDArray if necessary
            if isinstance(value, np.ndarray):
                node_to_shape_map[node] = ndarray.array(value, ctx=self.ctx).shape
            elif isinstance(value, (ndarray.NDArray, ndarray.ND_Sparse_Array)):
                node_to_shape_map[node] = value.shape
            else:
                assert False, "feed_dict value type not supported"
        
        for node in self.topo_order:
            if node not in node_to_shape_map:
                input_shapes = [node_to_shape_map[n] for n in node.inputs]
                node_to_shape_map[node] = node.op.infer_shape(
                        node, input_shapes)
        
        self.node_to_shape_map = node_to_shape_map 

        # for node in self.topo_order:
        #     if node  in feed_dict:
        #         continue
        #     input_vals = [ndarray.empty(shape = node_to_shape_map[n], ctx = self.ctx) for n in node.inputs]
        #     node_val = ndarray.empty(shape = node_to_shape_map[node], ctx = self.ctx)
        #     node.op.profile(node, input_vals, node_val, is_static = False)
        #     node.op.profile(node, input_vals, node_val, is_static = False)
        #     if is_print == True:
        #         profiler.PrintProfiler(node.name, node.profiler)
        #     # node.op.compute(node, input_vals, node_val, use_numpy = False)
        #     del input_vals
        #     del node_val

    def node_to_memory(self, node):
        shape = self.node_to_shape_map[node]
        tmp = 1.0
        for i in list(shape):
            tmp *= i
        tmp = tmp * 4 / (2**20)
        return tmp  
          
    def run(self, feed_dict = {}, convert_to_numpy_ret_vals = False):

        self.profile(feed_dict = feed_dict, is_print = True)
        used = 0
        for node in self.topo_order:
            used += self.node_to_memory(node)

        print("total_memory_usage: {} MB".format(used))

##################
# Helper Methods #
##################

def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)

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

def find_topo_sort2(node_list):
    total_node = find_topo_sort(node_list)
    input_degree = {}
    for node in total_node:
        input_degree[node] = 0
    for node in total_node:
        for n in node.inputs:
            input_degree[node] += 1
    from Queue import Queue
    ans = []
    q = Queue(maxsize = 0)
    for node in total_node:
        if input_degree[node] == 0:
            q.put(node)

    while q.qsize() > 0:
        thisNode = q.get()
        ans.append(thisNode)
        for node in total_node:
            if thisNode in node.inputs:
                input_degree[node] -= 1
                if input_degree[node] == 0:
                    q.put(node)
    return list(ans)

def sum_node_list(node_list):
    """Custom sum func to avoid creating redundant nodes in Python sum func."""
    node_list = [n for n in node_list if n is not None]
    if node_list == []:
        return None

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

def rewrite_topoorder(topo_order):
    for idx, node in enumerate(topo_order):
        if isinstance(node.op, Conv2d_Gradient_of_FilterOp) and idx != len(topo_order) - 1:
            tmp_node = topo_order[idx-1]
            topo_order[idx-1] = topo_order[idx]
            topo_order[idx] = tmp_node


