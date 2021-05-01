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
from .MatrixMult import MatMulOp
from .. import gpu_memory_manager
import ctypes
import os
from pynvml import *


class vdnnExecutor(object):
    """Executor computes values for given set of nodes in computation graph."""

    def __init__(self, eval_node_list, ctx = None, streams = None, policy = "all"):
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
        self.node_to_cpu_arr_map = None
        self.feed_shapes = None
        self.node_to_event_map = None
        self.outputNode2forwardNode = {}
        # swap: True, when a layer offloads its input feature map.
        #            False
        self.swap = {}

        # prefetched: True, this layer has already prefetched.
        #             False. 
        self.prefetched = {}
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

    def findPrefetchNode(self, currNodeId):
        # search all preceding layers
        for node in self.topo_order[currNodeId:]:
            for input_node in node.inputs:
                if self.swap[input_node] == True and self.node_to_arr_map[input_node].in_gpu == False:
                    # print("Prefetch {}, index {}".format(input_node, self.topo_order.index(input_node)))
                    return 1, input_node
                elif isinstance(input_node.op, Conv2dOp):
                    return -1, None
        return -1, None
        
    def set_vdnn_policy(self, policy):
        assert (self.ctx is not None)
        # init for each tensor
        for node in self.topo_order:
            self.swap[node] == False
        # for vdnn_all
        if policy == "all":
            from .Variable import PlaceholderOp
            for node in self.topo_order:
                if self.outputNode2forwardNode[node] == True and \
                        not isinstance(node.op, PlaceholderOp):
                    for input_node in node.inputs:
                        if isinstance(input_node.op, PlaceholderOp) and \
                            input_node.name in ("X", "y_"):
                            self.swap[input_node] = True
                        elif not isinstance(input_node.op, PlaceholderOp):
                            self.swap[input_node] = True
        # for vdnn_conv
        elif policy == "conv":
            # keep placeholder op in GPU memory
            for node in self.topo_order:
                if self.outputNode2forwardNode[node] == True and isinstance(node.op, Conv2dOp):
                    self.swap[node.inputs[0]] = True
        # for no policy
        else:
            pass

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
           For tensors which will be swapped into CPU, allocate the CPU array.
        
        Parameters
        ----------
        feed_shapes: node->shapes mapping for feed_dict nodes.
        """
        from .Variable import PlaceholderOp
        for node, shape in self.node_to_shape_map.items():
            if isinstance(node.op, PlaceholderOp) and node.tensor_value is not None:
                arr = node.tensor_value
                if isinstance(arr, np.ndarray) or (isinstance(arr, ndarray.NDArray) and not ndarray.is_gpu_ctx(arr.ctx)):
                    arr = ndarray.array(arr, ctx=self.ctx)
                    node.tensor_value = arr
                self.node_to_arr_map[node] = arr
            else:
                self.node_to_arr_map[node] = None
                if self.swap[node] is True:
                    self.node_to_cpu_arr_map[node] = ndarray.empty(shape)

    def node_to_memory(self, node):
        shape = self.node_to_shape_map[node]
        tmp = 1.0
        for i in list(shape):
            tmp *= i
        tmp = tmp * 4 / (2**20)
        return tmp #MB

    def shape_to_memory(self, shape):
        tmp = 1.0
        for i in list(shape):
            tmp *= i
        tmp = tmp * 4 / (2**20)
        return tmp

    def show_op_memory_usage(self, node):
        print("output node:", node.name, "Memory require: ", self.shape_to_memory(self.node_to_shape_map[node]), "node index:", self.topo_order.index(node),\
                "forward delete time", self.earlist_delete_time_forward[node], "backward delete time", self.earlist_delete_time_backward[node])
        for input_node in node.inputs:
            print("input node:", input_node.name,"Memory require: ", self.shape_to_memory(self.node_to_shape_map[input_node]),\
                "forward delete time", self.earlist_delete_time_forward[input_node], "backward delete time", self.earlist_delete_time_backward[input_node])
      
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
            self.node_to_cpu_arr_map = {}
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            if (not use_numpy):
                # vdnn policy
                self.set_vdnn_policy(self.policy)
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
            # Output tensors are not in GPU.
            if self.node_to_arr_map[node] == None or self.node_to_arr_map[node].in_gpu == False:
                if gpu_memory_manager.judge_malloc(self.node_to_shape_map[node]) == 0:
                    self.training_exit()
                if self.node_to_arr_map[node] == None:
                    node_val = ndarray.empty(self.node_to_shape_map[node], ctx = self.ctx)
                else:
                    self.node_to_arr_map[node].malloc_itself()
                    self.node_to_arr_map[node].in_gpu = True
                    node_val = self.node_to_arr_map[node]
            else:
                node_val = self.node_to_arr_map[node]

            # For forward node.
            if self.outputNode2forwardNode[node] == True:
                for input_node in node.inputs:
                    # this op/tensor need to be swapped and no longer use in the forward pass
                    if self.swap[input_node] == True and self.earlist_delete_time_forward[input_node] == node_index:
                        # print("Offload {}, index {} ".format(input_node.name, self.topo_order.index(input_node)))
                        self.node_to_cpu_arr_map[input_node].async_d2h(self.node_to_arr_map[input_node], self.streams[2], self.node_to_arr_map[input_node].gpu2cpu_event)
                        '''Free this NDArray'''
                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                for val in input_vals:
                    assert input_vals.in_gpu == True
                node.op.compute(node, input_vals, node_val, use_numpy, self.streams[0])
                
                self.streams[2].sync()
                self.streams[0].sync()
                self.node_to_arr_map[node] = node_val

                for input_node in node.inputs:
                    if self.swap[input_node] == True and self.earlist_delete_time_forward[input_node] == node_index:
                        self.node_to_arr_map[input_node].delete_itself()
                        
            # For backward node.
            else:
                # For input tensors which not in GPU
                for input_node in node.inputs:
                    if self.swap[input_node] == True and self.node_to_arr_map[input_node].in_gpu == False:
                        # print("Prefetch {}, index {} ".format(input_node.name, self.topo_order.index(input_node)))
                        if gpu_memory_manager.judge_malloc(self.node_to_shape_map[input_node]) == 0:
                            self.training_exit()
                        self.node_to_arr_map[input_node].malloc_itself()
                        self.node_to_arr_map[input_node].async_h2d(self.node_to_cpu_arr_map[input_node], self.streams[1],\
                                self.node_to_arr_map[input_node].cpu2gpu_event)
                        self.node_to_arr_map[input_node].in_gpu = True
                        self.streams[1].sync()

                # check wether need to prefetch! it will enlarge the max-batch size with performance degradation 
                ok, prefetchNode = self.findPrefetchNode(node_index)
                if ok != -1:
                    if gpu_memory_manager.judge_malloc(self.node_to_shape_map[prefetchNode]) == 0:
                        self.training_exit()
                    self.node_to_arr_map[prefetchNode].malloc_itself()
                    self.node_to_arr_map[prefetchNode].async_h2d(self.node_to_cpu_arr_map[prefetchNode],\
                                                                self.streams[1],\
                                                                self.node_to_arr_map[prefetchNode].cpu2gpu_event)
                    # print("Prefetch {}, index {} ".format(prefetchNode.name, self.topo_order.index(prefetchNode)))                

                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                for val in input_vals:
                    assert input_vals.in_gpu == True                    
                node.op.compute(node, input_vals, node_val, use_numpy, self.streams[0]) 
                self.streams[1].sync()
                self.streams[0].sync()
                self.node_to_arr_map[node] = node_val

                if ok != -1:
                    self.node_to_arr_map[prefetchNode].in_gpu = True
                
                for input_node in node.inputs:
                    if self.earlist_delete_time_backward[input_node] == node_index and\
                        (not isinstance(input_node.op, PlaceholderOp) or input_node.name in ("X", "y_")):
                        # print("free: ", input_node.name)
                        self.node_to_arr_map[input_node].delete_itself()
                
                if self.earlist_delete_time_backward[node] == -1 and\
                    not isinstance(node.op, PlaceholderOp):
                    self.node_to_arr_map[node].delete_itself()             
        
        for stream in self.streams:
            stream.sync()
            
        return [self.node_to_arr_map[n] for n in self.eval_node_list]
    
    def training_exit(self):
        assert 1 == -1, "GPU available memory is not enough!"
    
    def show_tensor_in_gpu(self):
        for node in self.topo_order:
            if self.node_to_arr_map[node] != None:
                if self.node_to_arr_map[node].in_gpu == True:
                    print(node.name, self.node_to_memory(node), self.topo_order.index(node), self.earlist_delete_time_forward[node], self.earlist_delete_time_backward[node])

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


