# -*- coding: utf-8 -*-
""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import
import numpy as np
from .Node import Op
from .. import ndarray
from ..stream import *
from ..logger import Logger
from .._base import get_array_memory
from .Variable import PlaceholderOp
from .Conv2d import Conv2dOp, Conv2d_Gradient_of_DataOp, Conv2d_Gradient_of_FilterOp
from .SoftmaxCrossEntropy import SoftmaxCrossEntropyOp, SoftmaxCrossEntropyGradientOp
from .OnesLike import OnesLikeOp
from .ZerosLike import ZerosLikeOp
from .MatrixMult import MatMulOp
from .. import gpu_memory_manager
import ctypes
import os
from pynvml import *

class recomputeExecutor(object):
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
        from .._base import _LIB
        _LIB.cudnn_init(ctypes.c_int(self.ctx.device_id), None)
        _LIB.cublas_init(ctypes.c_int(self.ctx.device_id), None)
        _LIB.cusp_init(ctypes.c_int(self.ctx.device_id), None)
        if streams is None:
            self.streams = []
            # compute stream
            self.streams.append(create_stream_handle(ctx))
            # h2d stream
            self.streams.append(create_stream_handle(ctx))
            # d2h stream
            self.streams.append(create_stream_handle(ctx))
        else:
            assert (len(streams) == 3)
            self.streams = streams
        for stream in self.streams:
            stream.sync()
        self.topo_order = find_topo_sort2(self.eval_node_list)
        rewrite_topoorder(self.topo_order)
        self.node_to_shape_map = None
        self.node_to_arr_map = None

        self.feed_shapes = None

        self.outputNode2forwardNode = {}
        self.checkpoints = {}
        self.get_dependency()
        self.policy = policy

    def get_dependency(self):
        self.earlist_delete_time_forward = {}  # the last use in the forward phase.
        self.earlist_delete_time_backward = {}  # the first time use in the backward phase.
        flag = 0
        for node in self.topo_order:
            if flag == 0:
                self.outputNode2forwardNode[node] = True
            else:
                self.outputNode2forwardNode[node] = False
            if isinstance(node.op, SoftmaxCrossEntropyOp):
                flag = 1 

        for node in self.topo_order:
            self.earlist_delete_time_forward[node] = -1
            self.earlist_delete_time_backward[node] = -1
        
        for idx, node in enumerate(self.topo_order):
            if isinstance(node.op, PlaceholderOp):
                continue
            for input_node in node.inputs:
                if self.outputNode2forwardNode[node] == True:
                    self.earlist_delete_time_forward[input_node] = max(idx, self.earlist_delete_time_forward[input_node])
                else:
                    self.earlist_delete_time_backward[input_node] = max(idx, self.earlist_delete_time_backward[input_node])

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

        for node in feed_shapes:
            self.node_to_shape_map[node] = feed_shapes[node]
        for node in self.topo_order:
            if node not in feed_shapes:
                input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
                self.node_to_shape_map[node] = node.op.infer_shape(
                    node, input_shapes)
    
    def check_in_candidate_split_point(self, node):
        if self.outputNode2forwardNode[node] == True:
            return True
        else:
            return False

    def memoryPlanningwithBudget(self, B = None):
        """
            param: 
                B: approximate memory budget. We can search over B to optimize the memory
                    allocation.
        """
        # Basic policy:
        # for node in self.topo_order:
        #     if isinstance(node.op, Conv2dOp) or isinstance(node.op, (PlaceholderOp, ZerosLikeOp)):
        #         self.checkpoints[node] = True
        #     else:
        #         self.checkpoints[node] = False
        # Search policy:
        B = 0 # try to find x and y
        tmp, x, y = 0, 0, 0
        for node in self.topo_order:
            tmp += self.node_to_memory(node)
            if isinstance(node.op, PlaceholderOp):
                self.checkpoints[node] = True
                continue
            if self.check_in_candidate_split_point(node) and tmp > B:
                x += self.node_to_memory(node)
                y = max(y, tmp)
                tmp = 0
                self.checkpoints[node] = True
            else:
                self.checkpoints[node] = False
        import math 
        bound = math.sqrt(x * y) # given by Page 12, "Search over Budget B" in Tianqi Chen's paper
        # for node in self.topo_order:
        #     print(node.name, self.checkpoints[node], self.node_to_memory(node))   
        # print("***" * 50)     
        B = bound
        tmp, x, y = 0, 0, 0
        for node in self.topo_order:
            tmp += self.node_to_memory(node)
            if isinstance(node.op, PlaceholderOp):
                self.checkpoints[node] = True
                continue
            if self.check_in_candidate_split_point(node) and tmp > B:
                x += self.node_to_memory(node)
                y = max(y, tmp)
                tmp = 0
                self.checkpoints[node] = True
            else:
                self.checkpoints[node] = False
        # for node in self.topo_order:
        #     print(node.name, self.checkpoints[node], self.node_to_memory(node))   

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        from .Variable import PlaceholderOp
        
        for node, shape in self.node_to_shape_map.items():
            # add for optimizer, PlaceholderOp with values directly handled
            if isinstance(node.op, PlaceholderOp) and node.tensor_value is not None:
                arr = node.tensor_value
                if isinstance(arr, np.ndarray) or (isinstance(arr, ndarray.NDArray) and not ndarray.is_gpu_ctx(arr.ctx)):
                    arr = ndarray.array(arr, ctx=self.ctx)
                    node.tensor_value = arr
                self.node_to_arr_map[node] = arr
            else:
                self.node_to_arr_map[node] = None
    
    def recompute_input_node_in_speed_mode(self, to_recompute_node):
        recompute_topo_order = []
        visited_set = set()

        # Change the recompute graph into topo
        def recomputation_dfs(recompute_node, visited = None, recompute_list = None):
            # For CheckPoint Node
            if self.node_to_arr_map[recompute_node].in_gpu == True\
                    or recompute_node in visited:
                return
            visited.add(recompute_node)
            for input_node in recompute_node.inputs:
                recomputation_dfs(input_node, visited_set, recompute_list)
            recompute_list.append(recompute_node)
        
        recomputation_dfs(to_recompute_node, visited_set, recompute_topo_order)
        for node in recompute_topo_order:
            # try to malloc the output tensor space of node.
            if gpu_memory_manager.judge_malloc(self.node_to_shape_map[node]) == 0:
                return False
            self.node_to_arr_map[node].malloc_itself()
            self.node_to_arr_map[node].in_gpu = True

            node_val = self.node_to_arr_map[node]            
            input_vals = [self.node_to_arr_map[n] for n in node.inputs]
            node.op.compute(node, input_vals, node_val, False, self.streams[0]) 
            self.streams[0].sync()
            self.node_to_arr_map[node] = node_val
        return True

    def recompute_input_node_in_memory_mode(self, to_recompute_node):
        recompute_topo_order = []
        visited_set = set()
        # Change the recompute graph into topo
        def recomputation_dfs(recompute_node, visited = None, recompute_list = None):
            # For CheckPoint Node
            if self.node_to_arr_map[recompute_node].in_gpu == True\
                    or recompute_node in visited:
                return
            visited.add(recompute_node)
            for input_node in recompute_node.inputs:
                recomputation_dfs(input_node, visited_set, recompute_list)
            recompute_list.append(recompute_node)
        
        recomputation_dfs(to_recompute_node, visited_set, recompute_topo_order)
        # get the dependency in the recompute_order
        
        dependency_node = {}
        for node in recompute_topo_order:
            dependency_node[node] = 0

        for node in recompute_topo_order:
            for input_node in node.inputs:
                if input_node in dependency_node:
                    dependency_node[input_node] += 1

        for node in recompute_topo_order:
            # try to malloc the output tensor space of node.
            if gpu_memory_manager.judge_malloc(self.node_to_shape_map[node]) == 0:
                return False
            self.node_to_arr_map[node].malloc_itself()
            self.node_to_arr_map[node].in_gpu = True

            node_val = self.node_to_arr_map[node]            
            input_vals = [self.node_to_arr_map[n] for n in node.inputs]

            node.op.compute(node, input_vals, node_val, False, self.streams[0]) 
            self.streams[0].sync()
            self.node_to_arr_map[node] = node_val
            for input_node in node.inputs:                
                if input_node in dependency_node:
                    dependency_node[input_node] -= 1
                    if dependency_node[input_node] == 0:
                        self.node_to_arr_map[input_node].delete_itself()
        return True

    def node_to_memory(self, node):
        shape = self.node_to_shape_map[node]
        tmp = 1.0
        for i in list(shape):
            tmp *= i
        tmp = tmp * 4 / (2**20)
        return tmp #MB

    def show_tensor_in_gpu(self):
        print("tensor still in gpu:")
        for gpunode in self.topo_order:
            if gpunode in self.node_to_arr_map and self.node_to_arr_map[gpunode] != None:
                if self.node_to_arr_map[gpunode].in_gpu == True:
                    print(gpunode.name, self.node_to_memory(gpunode))

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
            
        # Assume self.ctx is None implies numpy array and numpy ops.

        use_numpy = self.ctx is None
    
        feed_shapes = {}
        for node, value in feed_dict.items():
            feed_shapes[node] = value.shape

        if(not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.node_to_arr_map = {}
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            self.memoryPlanningwithBudget(24 * 1024)
            self.memory_plan(self.feed_shapes)

        for node, value in feed_dict.items():
            # convert values to ndarray.NDArray if necessary
            if isinstance(value, np.ndarray):
                self.node_to_arr_map[node] = ndarray.array(value, ctx=self.ctx)
            else:
                self.node_to_arr_map[node] = value
        for node_index, node in enumerate(self.topo_order):
            if node in feed_dict:
                continue
            # print("running node: ", node.name, " index: ",node_index)
            if self.node_to_arr_map[node] == None or self.node_to_arr_map[node].in_gpu == False:
                # print("malloc here")
                if gpu_memory_manager.judge_malloc(self.node_to_shape_map[node]) == 0:
                    # self.show_tensor_in_gpu()
                    self.training_exit()
                if self.node_to_arr_map[node] == None:
                    # print("malloc here1")
                    node_val = ndarray.empty(self.node_to_shape_map[node], ctx = self.ctx)
                else:
                    # print("malloc here2")
                    self.node_to_arr_map[node].malloc_itself()
                    self.node_to_arr_map[node].in_gpu = True
                    node_val = self.node_to_arr_map[node]
                # print("malloc ok")
            else:
                node_val = self.node_to_arr_map[node]

            if self.outputNode2forwardNode[node] == True:

                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                node.op.compute(node, input_vals, node_val, use_numpy, self.streams[0])
                self.streams[0].sync()
                self.node_to_arr_map[node] = node_val

                for input_node in node.inputs:
                    # drop the input tensor
                    if self.checkpoints[input_node] == False and\
                            self.earlist_delete_time_forward[input_node] == node_index:
                        self.node_to_arr_map[input_node].delete_itself()                
            # recompute in backward pass
            else:
                for input_node in node.inputs:
                    # need to recompute the input tensor
                    if self.node_to_arr_map[input_node].in_gpu == False:
                        # No enough space for recompute
                        """Speed Mode"""
                        if self.policy == "speed":
                            if self.recompute_input_node_in_speed_mode(input_node) == False:
                                self.training_exit()
                        elif self.policy == "memory":
                            """Memory Mode"""
                            if self.recompute_input_node_in_memory_mode(input_node) == False:
                                self.training_exit()
                                   
                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                node.op.compute(node, input_vals, node_val, use_numpy, self.streams[0]) 
                self.streams[0].sync()
                self.node_to_arr_map[node] = node_val

                for input_node in node.inputs:
                    if self.earlist_delete_time_backward[input_node] == node_index and\
                        (not isinstance(input_node.op, PlaceholderOp)):
                        self.node_to_arr_map[input_node].delete_itself()
                
                if self.earlist_delete_time_backward[node] == -1 and\
                    not isinstance(node.op, PlaceholderOp):
                    self.node_to_arr_map[node].delete_itself()
        
        for stream in self.streams:
            stream.sync()
        if not use_numpy and convert_to_numpy_ret_vals:
            return [self.node_to_arr_map[n].asnumpy() for n in self.eval_node_list]
        return [self.node_to_arr_map[n] for n in self.eval_node_list]
    
    def training_exit(self):
        assert 1 == -1, "GPU available memory is not enough!"

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

def rewrite_topoorder(topo_order):
    for idx, node in enumerate(topo_order):
        if isinstance(node.op, Conv2d_Gradient_of_FilterOp) and idx != len(topo_order) - 1:
            tmp_node = topo_order[idx-1]
            topo_order[idx-1] = topo_order[idx]
            topo_order[idx] = tmp_node


