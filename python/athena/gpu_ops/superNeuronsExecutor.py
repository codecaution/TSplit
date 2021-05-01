# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from .Node import Op
from .. import ndarray
from ..stream import *
from ..logger import Logger
from .._base import get_array_memory
from athena.gpu_ops.Conv2d import Conv2dOp, Conv2d_Gradient_of_DataOp, Conv2d_Gradient_of_FilterOp
from athena.gpu_ops.SoftmaxCrossEntropy import SoftmaxCrossEntropyOp, SoftmaxCrossEntropyGradientOp
from athena.gpu_ops.Variable import PlaceholderOp
from athena.gpu_ops.ZerosLike import ZerosLikeOp
from athena.gpu_ops.MatrixMult import MatMulOp
from athena.gpu_ops.Relu import ReluOp
from athena.gpu_ops.MaxPool import Max_Pool2dOp
import ctypes
import os
from .. import gpu_memory_manager

# LRU cache for swapped tensors.
# A tensor can't be removed from cache if locked.
# A locked tensor means it is used for calculation.
class LRUCache(object):
    
    def __init__(self):
        self.LRU_tensor_list = list()
        self.tensor_lock = dict()
        self.tensor_offloaded = dict()
        self.tensor_index = dict() # just for debug.

    def lock_tensor(self, tensor):
        self.tensor_lock[tensor] = True

    def unlock_tensor(self, tensor):
        self.tensor_lock[tensor] = False

    # Insert a tensor to its front.
    def insert(self, tensor, idx = None):
        self.tensor_lock[tensor] = False
        self.LRU_tensor_list.insert(0, tensor)
        self.tensor_offloaded[tensor] = False
        if idx != None:
            self.tensor_index[tensor] = idx
    
    def delete(self, tensor):
        self.LRU_tensor_list.remove(tensor)
        self.tensor_lock.pop(tensor)
        tensor.delete_itself()

    # remove enough bytes for new tensors.
    def out(self, tensor_shape):
        # free_memory = 0
        while gpu_memory_manager.judge_malloc(tensor_shape) == 0:
            # need to free some tensor before.
            free_tensor  = self._getLastUnlockedTensor()
            if free_tensor == None:
                self.training_exit()
            if self.tensor_offloaded[free_tensor] == False:
                free_tensor.gpu2cpu_event.sync()
                self.tensor_offloaded[free_tensor] = True
            self.delete(free_tensor)
            # print("delete tensor", free_tensor.shape)
    
    # check given tensor in gpu or not.
    def check(self, tensor, idx = None):
        isFound = (tensor in self.LRU_tensor_list)
        if isFound == False:
            if gpu_memory_manager.judge_malloc(tensor.shape) == 0:
                self.out(tensor.shape)
            tensor.malloc_itself()
            self.insert(tensor, idx)
        else:
            self._placeToFront(tensor)
        return isFound

    def _placeToFront(self, tensor):
        assert tensor in self.LRU_tensor_list
        self.LRU_tensor_list.remove(tensor)
        self.LRU_tensor_list.insert(0, tensor)
    
    def _getLastUnlockedTensor(self):
        # print("tensor list length:", len(self.LRU_tensor_list))
        for i in range(len(self.LRU_tensor_list) - 1, -1, -1):
            tensor = self.LRU_tensor_list[i]
            if self.tensor_lock[tensor] == False:
                return tensor
        return None

    def _show_tensor_list(self):
        print("show LRU Cache List")
        for i in range(len(self.LRU_tensor_list)):
            tensor = self.LRU_tensor_list[i]
            # print(tensor.shape, self.tensor_lock[tensor], self.tensor_index[tensor])

    def training_exit(self):
        assert 1 == -1, "GPU available memory is not enough!"

class superNeuronsExecutor(object):
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
            # computation stream
            self.streams.append(create_stream_handle(ctx))
            # h2d stream
            self.streams.append(create_stream_handle(ctx))
            # d2h stream
            self.streams.append(create_stream_handle(ctx))
        
        for stream in self.streams:
            stream.sync()
        self.topo_order = find_topo_sort2(self.eval_node_list)
        rewrite_topoorder(self.topo_order)
        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.node_to_cpu_arr_map = None
        self.feed_shapes = None

        self.outputNode2forwardNode = {}
        # swap: True, when a layer offloads its input feature map.
        #            False
        self.swap = {}
        self.prefetch_list = []
        # prefetched: True, this layer has already prefetched.
        #             False. 
        self.recompute = {}
        self.get_dependency()

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

    def superneurons_policy(self, execution_plan = None):
        """
            We design and implement SuperNeurons to enable DL practitioners to explore deep neural networks; 
                and the largest computable network of SuperNeurons is only bounded by the maximum memory usage among layers.
            
            Policy:
                checkpoints: compute-intensive(CONV).
                LRU Tensor Cache to reduce communications.
                Liveness Analysis, Unified Tensor Pool, and Cost-Aware Recomputation together to use the max memory as the Max(layer_i)
        """
        for node in self.topo_order:
            self.swap[node] = False
            self.recompute[node] = False
        if execution_plan == None:
            for node in self.topo_order:
                if self.outputNode2forwardNode[node]:
                    # Swap the output tensor of Conv 
                    if isinstance(node.op, Conv2dOp):
                        self.swap[node] = True
                    # Recompute other tensor
                    elif not isinstance(node.op, (PlaceholderOp, ZerosLikeOp)):
                        self.recompute[node] = True
        else:
            for node in self.topo_order:
                if execution_plan.swap[node] == True:
                    self.swap[node] = True
                elif execution_plan.recompute[node] == True:
                    self.recompute[node] == True

    def memory_plan(self, feed_shapes):
        """Allocates ndarray.NDArray for every node except feed_dict nodes.
           For tensors which will be swapped into CPU, allocate the CPU array.
        
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
                if self.swap[node] is True:
                    self.node_to_cpu_arr_map[node] = ndarray.empty(shape)
    
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

    # speed mode: store the intermediate feature map
    def recompute_input_node_in_speed_mode(self, to_recompute_node, tensor_cache):
        recompute_topo_order = []
        visited_set = set()

        # Change the recompute graph into topo
        def recomputation_dfs(recompute_node, visited = None, recompute_list = None):
            # For CheckPoint Node
            if self.node_to_arr_map[recompute_node].in_gpu == True\
                    or self.swap[recompute_node] == True or recompute_node in visited:
                return
            visited.add(recompute_node)
            for input_node in recompute_node.inputs:
                recomputation_dfs(input_node, visited_set, recompute_list)
            recompute_list.append(recompute_node)
        
        # Get the recompute list
        recomputation_dfs(to_recompute_node, visited_set, recompute_topo_order)
        
        # In the recompute process: 
        for node in recompute_topo_order:
            # try to malloc the output tensor space of node.
            if self.swap[node] == True:
                if self.prefetched[node] == False:
                    tensor_cache.check(self.node_to_arr_map[node], self.topo_order.index(node))
                    self.node_to_arr_map[node].async_h2d(self.node_to_cpu_arr_map[node],\
                            self.streams[1], self.node_to_arr_map[node].cpu2gpu_event) 
                    tensor_cache.lock_tensor(self.node_to_arr_map[node])
                    self.prefetched[node] = True
                self.node_to_arr_map[node].cpu2gpu_event.sync()
                self.node_to_arr_map[node].in_gpu = True
                continue
            else:
                if gpu_memory_manager.judge_malloc(self.node_to_shape_map[node]) == 0:
                    tensor_cache.out(self.node_to_shape_map[node])
                # set the out of cache tensor to be prefetch = False
                self.adjust_prefetch_list(tensor_cache)
                self.node_to_arr_map[node].malloc_itself()


            node_val = self.node_to_arr_map[node]
            input_vals = [self.node_to_arr_map[n] for n in node.inputs]
            for input_node in node.inputs:
                assert self.node_to_arr_map[input_node].in_gpu == True
            node.op.compute(node, input_vals, node_val, False, self.streams[0]) 
            self.streams[0].sync()
            self.node_to_arr_map[node] = node_val
        return True

    # memory mode: not store the intermediate feature map
    def recompute_input_node_in_memory_mode(self, to_recompute_node, tensor_cache):
        recompute_topo_order = []
        visited_set = set()

        # Change the recompute graph into topo
        def recomputation_dfs(recompute_node, visited = None, recompute_list = None):
            # For CheckPoint Node
            if self.node_to_arr_map[recompute_node].in_gpu == True\
                    or recompute_node in visited:
                return

            visited.add(recompute_node)
            if self.swap[recompute_node] == False:
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
            if self.swap[node] == True:
                if self.prefetched[node] == False:
                    tensor_cache.check(self.node_to_arr_map[node], self.topo_order.index(node))
                    self.node_to_arr_map[node].async_h2d(self.node_to_cpu_arr_map[node],\
                            self.streams[1], self.node_to_arr_map[node].cpu2gpu_event) 
                    tensor_cache.lock_tensor(self.node_to_arr_map[node])
                    self.prefetched[node] = True
                self.node_to_arr_map[node].cpu2gpu_event.sync()
                self.node_to_arr_map[node].in_gpu = True
                continue
            else:
                if gpu_memory_manager.judge_malloc(self.node_to_shape_map[node]) == 0:
                    tensor_cache.out(self.node_to_shape_map[node])
                self.adjust_prefetch_list(tensor_cache)
                self.node_to_arr_map[node].malloc_itself()

            node_val = self.node_to_arr_map[node]            
            input_vals = [self.node_to_arr_map[n] for n in node.inputs]

            for input_node in node.inputs:
                assert self.node_to_arr_map[input_node].in_gpu == True

            node.op.compute(node, input_vals, node_val, False, self.streams[0]) 
            self.streams[0].sync()
            self.node_to_arr_map[node] = node_val

            # delete the unused input tensor according to the dependency graph
            for input_node in node.inputs:                
                if input_node in dependency_node:
                    dependency_node[input_node] -= 1
                    if dependency_node[input_node] == 0:
                        if self.swap[input_node] == True:
                            # print("delete ", input_node.name, self.node_to_arr_map[input_node].shape, self.topo_order.index(input_node))
                            tensor_cache.unlock_tensor(self.node_to_arr_map[input_node])
                        else:
                            # For Recompute Op
                            self.node_to_arr_map[input_node].delete_itself()
        return True

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
        
    def adjust_prefetch_list(self, cache):
        for node in self.prefetch_list:
            tensor = self.node_to_arr_map[node]
            if tensor not in cache.LRU_tensor_list:
                self.prefetched[node] = False

    def run(self, feed_dict, convert_to_numpy_ret_vals=False):
        """
        Parameters
        ----------
        feed_dict: a dictionary of node->np.ndarray supplied by user.
        convert_to_numpy_ret_vals: whether to convert ret vals to np.array.

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
            self.superneurons_policy()
            self.memory_plan(self.feed_shapes)
        
        # add the array of params, X and Y_ to node_to_arr_map
        for node, value in feed_dict.items():
            if isinstance(value, np.ndarray):
                self.node_to_arr_map[node] = ndarray.array(value, ctx = self.ctx)
            else:
                self.node_to_arr_map[node] = value

        self.offloaded = {}
        self.prefetched = {}
        for node in self.topo_order:
            if self.swap[node] == True:
                self.offloaded[node] = False
                self.prefetched[node] = False

        tensor_cache = LRUCache()
        # D2H_stream = []  # Device to Host stream, when a tensor is beginned to swap-out, add it to this stream
        # H2D_stream = []  # Host to Device stream, when a tensor is beginned to swap-in, add it to this stream
        #                  # A tensor maybe swap-in multi-times. /
        
        for node_index, node in enumerate(self.topo_order):
            if node in feed_dict:
                continue
            
            # print("==="*50)
            # print("running ", node.name, self.topo_order.index(node))
            # gpu_memory_manager.print_gpu_memory()
            # self.show_op_memory_usage(node)

            # Malloc the output array
            # 换出阻塞计算
            # self.node_to_arr_map[node] == None: it is the first batch.
            #                          .in_gpu == False: it is deleted in the backward
            if self.node_to_arr_map[node] == None or self.node_to_arr_map[node].in_gpu == False:
                """ idle for memory demand"""
                if gpu_memory_manager.judge_malloc(self.node_to_shape_map[node]) == 0:
                    tensor_cache.out(self.node_to_shape_map[node])
                self.adjust_prefetch_list(tensor_cache)
                if self.node_to_arr_map[node] == None:
                    # print("malloc here")
                    node_val = ndarray.empty(self.node_to_shape_map[node], ctx = self.ctx)
                    self.node_to_arr_map[node] = node_val
                    self.node_to_arr_map[node].in_gpu = True
                else:
                    self.node_to_arr_map[node].malloc_itself()
                    node_val = self.node_to_arr_map[node]
            else:
                node_val = self.node_to_arr_map[node]
            # print("malloc ok")
            if self.swap[node] == True:
                self.prefetch_list.append(node)
            # lock the input tensor
            for input_node in node.inputs:
                if self.node_to_arr_map[input_node].in_gpu == True and self.swap[input_node] == True:
                    tensor_cache.lock_tensor(self.node_to_arr_map[input_node])

            # SwapOut in forward pass
            if self.outputNode2forwardNode[node] == True:
                # prefetch the input tensor, if it has been offload during the forward.
                for input_node in node.inputs:
                    if self.swap[input_node] == True:
                        tensor_cache.check(self.node_to_arr_map[input_node], self.topo_order.index(input_node))
                        if self.node_to_arr_map[input_node].in_gpu == False:
                            self.node_to_arr_map[input_node].async_h2d(self.node_to_cpu_arr_map[input_node],\
                                    self.streams[1], self.node_to_arr_map[input_node].cpu2gpu_event) 
                            self.node_to_arr_map[input_node].cpu2gpu_event.sync()
                            self.node_to_arr_map[input_node].in_gpu = True
                            tensor_cache.lock_tensor(self.node_to_arr_map[input_node])
                
                # 所有input val ok
                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                for input_node in node.inputs:
                    assert self.node_to_arr_map[input_node].in_gpu == True
                # gpu_memory_manager.print_gpu_used_memory("forward")
                node.op.compute(node, input_vals, node_val, use_numpy, self.streams[0])
                self.streams[0].sync()
                self.node_to_arr_map[node] = node_val

                # swapout the output
                if self.swap[node] == True:
                    tensor_cache.insert(self.node_to_arr_map[node], self.topo_order.index(node))
                    self.node_to_cpu_arr_map[node].async_d2h(self.node_to_arr_map[node], self.streams[2], self.node_to_arr_map[node].gpu2cpu_event)

                # drop the recompute node
                for input_node in node.inputs:
                    if self.earlist_delete_time_forward[input_node] == node_index:
                        if self.recompute[input_node] == True or self.earlist_delete_time_backward[input_node] == -1 and not isinstance(input_node.op, PlaceholderOp):
                            self.node_to_arr_map[input_node].delete_itself()
                
                # unlock tensor from tensor cache.
                for input_node in node.inputs:
                    if self.swap[input_node] == True:
                        tensor_cache.unlock_tensor(self.node_to_arr_map[input_node])

            # SwapIn in backward pass
            else:
                for input_node in node.inputs:
                    # on-demand swap in
                    # print("input node not in gpu", self.node_to_arr_map[input_node].in_gpu, self.swap[input_node])
                    if self.node_to_arr_map[input_node].in_gpu == False and\
                            self.swap[input_node] == True:
                        if self.prefetched[input_node] == False:
                            tensor_cache.check(self.node_to_arr_map[input_node], self.topo_order.index(input_node))
                            self.node_to_arr_map[input_node].async_h2d(self.node_to_cpu_arr_map[input_node],\
                                    self.streams[1], self.node_to_arr_map[input_node].cpu2gpu_event) 
                            # self.node_to_arr_map[input_node].sync()
                            # self.node_to_arr_map[input_node].in_gpu = True
                            tensor_cache.lock_tensor(self.node_to_arr_map[input_node])
                            self.prefetched[input_node] = True

                # recompute the input node 
                for input_node in node.inputs:
                    if self.node_to_arr_map[input_node].in_gpu == False and\
                        self.recompute[input_node] == True:
                        # self.recompute_input_node_in_speed_mode(input_node, tensor_cache)
                        self.recompute_input_node_in_memory_mode(input_node, tensor_cache)
                
                # sync the prefetch streams
                for input_node in node.inputs:
                    if self.node_to_arr_map[input_node].in_gpu == False and\
                        self.swap[input_node] == True and self.prefetched[input_node] == True:
                        self.node_to_arr_map[input_node].cpu2gpu_event.sync()
                        self.node_to_arr_map[input_node].in_gpu = True

                if isinstance(node.op, Conv2d_Gradient_of_FilterOp):
                    ok, prefetch_node = self.findPrefetchNode(node_index)
                    if ok == 1:
                        if gpu_memory_manager.judge_malloc(self.node_to_shape_map[prefetch_node]) == 0:
                            tensor_cache.out(self.node_to_arr_map[prefetch_node])
                        self.adjust_prefetch_list(tensor_cache)
                        self.prefetched[prefetch_node] = True
                        self.node_to_arr_map[prefetch_node].malloc_itself()
                        self.node_to_arr_map[prefetch_node].in_gpu = False
                        self.node_to_arr_map[prefetch_node].async_h2d(self.node_to_cpu_arr_map[prefetch_node],\
                                                                        self.streams[1],
                                                                        self.node_to_arr_map[prefetch_node].cpu2gpu_event)
                for input_node in node.inputs:
                    assert self.node_to_arr_map[input_node].in_gpu == True    

                input_vals = [self.node_to_arr_map[n] for n in node.inputs]
                for input_node in node.inputs:
                    assert self.node_to_arr_map[input_node].in_gpu == True
                # gpu_memory_manager.print_gpu_used_memory("backward")
                node.op.compute(node, input_vals, node_val, use_numpy, self.streams[0])
                self.streams[0].sync()
                self.node_to_arr_map[node] = node_val
                
                for input_node in node.inputs:
                    if self.swap[input_node] == True:
                        tensor_cache.unlock_tensor(self.node_to_arr_map[input_node])

                for input_node in node.inputs:
                    if self.earlist_delete_time_backward[input_node] == node_index and\
                        not isinstance(input_node.op, PlaceholderOp):
                        if self.swap[input_node] == True:
                            # print(input_node.name, self.node_to_arr_map[input_node].in_gpu, self.topo_order.index(input_node))
                            tensor_cache.delete(self.node_to_arr_map[input_node])
                            self.adjust_prefetch_list(tensor_cache)
                        self.node_to_arr_map[input_node].delete_itself()

                if self.earlist_delete_time_backward[node] == -1 and\
                    not isinstance(node.op, PlaceholderOp):
                    self.node_to_arr_map[node].delete_itself()

        # for stream in self.streams:
        #     stream.sync()
        if not use_numpy and convert_to_numpy_ret_vals:
            return [self.node_to_arr_map[n].asnumpy() for n in self.eval_node_list]
        return [self.node_to_arr_map[n] for n in self.eval_node_list]
    
    def show_policy(self):
        for node in self.topo_order:
            print(node.name, self.topo_order.index(node),"swap: ", self.swap[node],"recompute: ", self.recompute[node], self.node_to_memory(node))
    
    def node_to_memory(self, node):
        shape = self.node_to_shape_map[node]
        tmp = 1.0
        for i in list(shape):
            tmp *= i
        tmp = tmp * 4 / (2**20)
        return tmp
    
    def show_tensor_in_gpu(self):
        for node in self.topo_order:
            if self.node_to_arr_map[node] != None and not isinstance(node.op, (PlaceholderOp, ZerosLikeOp)):
                if self.node_to_arr_map[node].in_gpu == True:
                    print(node.name, self.node_to_memory(node), self.topo_order.index(node), self.earlist_delete_time_forward[node], self.earlist_delete_time_backward[node])
    
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