""" library to take autodiff and execute a computation graph """
from __future__ import absolute_import

import numpy as np
import scipy
from . import ndarray, gpu_op
import ctypes
import os
#from pynvml import *
NAME_RULE = 1
G_NODE_ID = 0
FLAG_SHOW_GRAPH = False

def communicate_init(worker_num, worker_id, source_ip, target_ip): 
    global lib_communicate
    # lib_communicate.DL_Connect_Init(2, 0, "*:4001", "localhost:4002")
    # lib_communicate.DL_Connect_Init(2, 1, "*:4002", "localhost:4001")
    curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(curr_path, '../../build/lib/')
    path_to_so_file = os.path.join(lib_path, "lib_communication.so")
    lib_communicate = ctypes.cdll.LoadLibrary(path_to_so_file)
    lib_communicate.DL_Connect_Init(worker_num, worker_id, source_ip, target_ip)
def communicate_finish():
    lib_communicate.DL_Communicate_Close()

def worker_init():
    global worker_communicate

    # subprocess.check_call('pwd')
    # subprocess.check_call(['/bin/ls', './ps_lib.sh'])
    # subprocess.check_call(['./ps_lib.sh', '1', '1', './test_kv_app'])

    # os.environ['DMLC_NUM_WORKER'] = "2"
    # os.environ['DMLC_NUM_SERVER'] = "5"
    worker_id=os.getenv("HBSUN_WORKER_ID")
    scheduler_ip=os.getenv("DMLC_PS_ROOT_URI")
    scheduler_port=os.getenv("DMLC_PS_ROOT_PORT")
    print("hbsun:", worker_id, scheduler_ip, scheduler_port)
    os.environ['DMLC_PS_ROOT_URI'] = os.getenv("DMLC_PS_ROOT_URI")
    os.environ['DMLC_PS_ROOT_PORT'] = os.getenv("DMLC_PS_ROOT_PORT")
    print("DMLC_PS_ROOT_URI=", os.environ['DMLC_PS_ROOT_URI'])
    print("DMLC_PS_ROOT_PORT=", os.environ['DMLC_PS_ROOT_PORT'])
    os.environ['DMLC_ROLE'] = "worker"
    os.environ['HEAPPROFILE'] = "./W"+worker_id
    # worker_id=os.getenv('HEAPPROFILE')
    # print ("worker_id=",worker_id,worker_id=='./W0')

    ll = ctypes.cdll.LoadLibrary
    worker_communicate = ll("../build/lib/worker.so")
    worker_communicate.DL_Init()
    worker_id=os.getenv('HEAPPROFILE')
    print ("worker_id=",worker_id," has launched successful")


def worker_finish():
    worker_communicate.DL_Finalize()


def server_init():
    global server_communicate
    # print 'how are you!!!'
    # sys.stdout.flush()
    # subprocess.check_call('pwd')
    # subprocess.check_call(['/bin/ls', './ps_lib.sh'])
    # subprocess.check_call(['./ps_lib.sh', '1', '1', './test_kv_app'])

    # os.environ['DMLC_NUM_WORKER'] = "2"
    # os.environ['DMLC_NUM_SERVER'] = "5"

    server_id=os.getenv("HBSUN_SERVER_ID")

    #os.environ['DMLC_PS_ROOT_URI'] = "127.0.0.1"
    #os.environ['DMLC_PS_ROOT_PORT'] = "8000"
    #os.environ['DMLC_ROLE'] = "worker"

    os.environ['HEAPPROFILE'] = "./S"+server_id

    # worker_id=os.getenv('HEAPPROFILE')
    # print ("worker_id=",worker_id,worker_id=='./W0')

    ll = ctypes.cdll.LoadLibrary
    server_communicate = ll("../build/lib/server.so")
    server_communicate.DL_Init()
    server_communicate.StartServer()

    server_id=os.getenv('HEAPPROFILE')
    print ("server_id=",server_id," has launched successful")


def server_finish():
    server_communicate.DL_Finalize()

def scheduler_init():
    global scheduler_communicate

    # os.environ['DMLC_NUM_WORKER'] = "2"
    # os.environ['DMLC_NUM_SERVER'] = "5"
    #worker_id=os.getenv("HBSUN_WORKER_ID")
    #os.environ['DMLC_PS_ROOT_URI'] = "127.0.0.1"
    #os.environ['DMLC_PS_ROOT_PORT'] = "8000"
    #os.environ['DMLC_ROLE'] = "worker"

    #os.environ['HEAPPROFILE'] = "./W"+worker_id

    # worker_id=os.getenv('HEAPPROFILE')
    # print ("worker_id=",worker_id,worker_id=='./W0')

    ll = ctypes.cdll.LoadLibrary
    scheduler_communicate = ll("../build/lib/scheduler.so")
    scheduler_communicate.DL_Init()
    #worker_id=os.getenv('HEAPPROFILE')
    #print ("worker_id=",worker_id," has launched successful")
    print ("scheduler  has launched successful")


def scheduler_finish():
    scheduler_communicate.DL_Finalize()



class Node(object):
    """Node in a computation graph."""
    def __init__(self):
        """Constructor, new node is indirectly created by Op object call method.

            Instance variables
            ------------------
            self.inputs: the list of input nodes.
            self.op: the associated op object,
                e.g. add_op if this node is created by adding two other nodes.
            self.const_attr: the add or multiply constant.
                e.g. self.const_attr=5 if this node is created by x+5.
            self.name: node name for debugging.
        """
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""
        self.desc = ""
        global G_NODE_ID
        self.id = G_NODE_ID
        G_NODE_ID = G_NODE_ID + 1
        self.swap = False

    def __add__(self, other):
        """Adding two nodes return a new node."""
        if isinstance(other, Node):
            new_node = add_op(self, other)
        else:
            # Add by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = add_byconst_op(self, other)
        return new_node

    def __mul__(self, other):
        """Multiplying two nodes return a new node."""
        if isinstance(other, Node):
            new_node = mul_op(self, other)
        else:
            # Mul by a constant stores the constant in new node's const_attr
            # 'other' argument is a constant
            new_node = mul_byconst_op(self, other)
        return new_node

    # Allow left-hand-side add and multiply.
    __radd__ = __add__
    __rmul__ = __mul__

    def __str__(self):
        """Allow print to display node name."""
        return self.name


class Op(object):
    """Op represents operations performed on nodes."""
    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_vals, output_val, use_numpy=True):
        """Given values of input nodes, compute the output value.

        Parameters
        ----------
        node: node that performs the compute.
        input_vals: values of input nodes.
        output_val: output value of the node, modified in-place.
        use_numpy: bool flag whether to use numpy for compute
        """
        raise NotImplementedError

    def gradient(self, node, output_grad):
        """Given output gradient, compute partial gradient to each input node.

        Parameters
        ----------
        node: node that performs the gradient.
        output_grad: output gradient summed from children nodes' contributions

        Returns
        -------
        A list of gradient contributions to each input node respectively.
        """
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        """Given shapes of input nodes, compute shape of output node.

        Implementation note:
        It's simpler to treat shape of constants as (1,), so that constants can
        be stored as a numpy array too and you would need fewer special case
        handling.

        Parameters
        ----------
        node: node whose shape is being inferred.
        input_vals: shapes of input nodes.

        Returns
        -------
        A tuple representing the shape of output node.
        """
        raise NotImplementedError


class Distributed_CommunicateOp(Op):
    def __call__(self,nodeA):
        new_node = Op.__call__(self)
        new_node.inputs = [nodeA]
        new_node.name = "Distributed_Communicate(%s)" % (nodeA.name)
        # print nodeA.name
        return new_node
    
    def compute(self,node,input_vals, output_val, use_numpy = True):
        after_reduce_gradient_cpu = ndarray.empty(shape = output_val.shape, ctx = ndarray.cpu(0))
        if use_numpy:
            gradient_val_cpu = ndarray.array(input_vals[0], ctx = ndarray.cpu(0))
        else:
            gradient_val_cpu = ndarray.array(input_vals[0].asnumpy(), ctx = ndarray.cpu(0))
        # print gradient_val_cpu.asnumpy()
        lib_communicate.DL_Communicate_Init(gradient_val_cpu.handle)
        lib_communicate.DL_Communicate(gradient_val_cpu.handle, after_reduce_gradient_cpu.handle)
        # print after_reduce_gradient_cpu.asnumpy()
        if use_numpy:
            output_val[:] = after_reduce_gradient_cpu.asnumpy()
        else:
            after_reduce_gradient_cpu.copyto(output_val)
    
    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]



class Ps_CommunicateOp(Op):
    def __call__(self, nodeA, learning_rate):
        
        new_node = Op.__call__(self)
        new_node.inputs = [nodeA]
        self.learning_rate = learning_rate
        new_node.name = "Ps_Communicate(%s)" % (nodeA.name)

        # print nodeA.name
        return new_node

    ## if phase == "train" return reduced gradient value ,if phase == "test", return gradient value , but is descarded
    def compute(self,node,phase,parameter_id, input_vals, output_val, use_numpy = None):
        after_reduce_gradient_cpu = ndarray.empty(shape = output_val.shape)

        if use_numpy:

            input_vals[0] = input_vals[0] * self.learning_rate

            gradient_val_cpu = ndarray.array(input_vals[0], ctx = ndarray.cpu(0))

        else:
            # print node.name
            # print input_vals[0].asnumpy()
            # print 'before mul', input_vals[0].asnumpy()
            gpu_op.matrix_elementwise_multiply_by_const(
                input_vals[0], self.learning_rate, input_vals[0])
            # print 'after mul', input_vals[0].asnumpy()
            # print input_vals[0].asnumpy()

            gradient_val_cpu = ndarray.array(input_vals[0].asnumpy(), ctx = ndarray.cpu(0))


        # lib_communicate.DL_Communicate_Init(gradient_val_cpu.handle)
        # worker_id=os.getenv("HBSUN_WORKER_ID")
        # print 'Ps_CommunicateOp :', worker_id, node.id-43 , node.name, gradient_val_cpu.shape

        if phase == "train":
          worker_communicate.DL_Communicate_BSP(parameter_id, gradient_val_cpu.handle, after_reduce_gradient_cpu.handle)
        # worker_communicate.DL_Communicate(node.id-43, gradient_val_cpu.handle, after_reduce_gradient_cpu.handle)
        # print after_reduce_gradient_cpu.asnumpy()
        # print after_reduce_gradient_cpu
        # if(node.id-43==2):
        #     print gradient_val_cpu.asnumpy()
        #     print after_reduce_gradient_cpu.asnumpy()
        #     print "xxx"
          if use_numpy:
            output_val[:] = after_reduce_gradient_cpu.asnumpy()
          else:
            after_reduce_gradient_cpu.copyto(output_val)
        else:
          if use_numpy:
            output_val[:] = gradient_val_cpu.asnumpy()
          else:
            gradient_val_cpu.copyto(output_val)


            

    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[0]










# Create global singletons of operators.
distributed_communicate_op = Distributed_CommunicateOp()
ps_communicate_op = Ps_CommunicateOp()



class Executor(object):
    """Executor computes values for given set of nodes in computation graph."""
    def __init__(self, eval_node_list, ctx=None, policy=None,init_ps=False):
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
        self.topo_order = find_topo_sort(self.eval_node_list)
        self.node_to_shape_map = None
        self.node_to_arr_map = None
        self.feed_shapes = None
        self.policy = policy
        self.init_ps = init_ps
        self.init_count =0
        if self.policy == 'swap':
          self.swap_queue = []
        self.worker_id=os.getenv('HEAPPROFILE')

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
        
        for node in self.topo_order:
          if node in feed_shapes:
            self.node_to_shape_map[node] = feed_shapes[node]
          else:
            # print(node.name)
            input_shapes = [self.node_to_shape_map[n] for n in node.inputs]
            self.node_to_shape_map[node] = node.op.infer_shape(node, input_shapes)
        

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
        assert (self.ctx is not None)
        #self.infer_shape(feed_shapes)
        self.node_to_arr_map = {}
        for node, shape in self.node_to_shape_map.items():
          if self.policy == 'swap':
            if not node.swap:
              self.node_to_arr_map[node] = ndarray.empty(shape, ctx=self.ctx)
          elif self.policy == 'vdnn':
            self.node_to_arr_map[node] = np.empty(shape)
          else:
            self.node_to_arr_map[node] = ndarray.empty(shape, ctx=self.ctx)


    ## phase is a string , "train" or "evaluate"
    def run(self, feed_dict, convert_to_numpy_ret_vals=False, phase="train", ps_map={}):

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
        node_to_val_map = {}

        for node, value in feed_dict.items():
            if use_numpy:
                # all values passed in feed_dict must be np.ndarray
                assert isinstance(value, (np.ndarray, scipy.sparse.spmatrix))
                
                if isinstance(value, np.ndarray):
                    if phase == "train":
                        value1 = ndarray.array(value, ctx = ndarray.cpu(0))
                        value2 = ndarray.empty(shape = value.shape, ctx = ndarray.cpu(0))

                        if self.init_ps and self.init_count<len(self.eval_node_list):
                            if self.worker_id!='./W0':
                                # print(self.worker_id, " no", self.init_count)
                                zero_array=np.zeros(value.shape)
                                value1 = ndarray.array(zero_array, ctx = ndarray.cpu(0))
                                # print value1.asnumpy()[:]
                            else:
                                pass
                                # print(self.worker_id, " yes", self.init_count)
                            worker_communicate.DL_Communicate_BSP(node.id, value1.handle, value2.handle)
                            self.init_count+=1

                    node_to_val_map[node] = value
                
                else:
                    # TODO: need to implement ps for sparse
                    
                    node_to_val_map[node] = value

            else:
                # convert values to ndarray.NDArray if necessary
                if isinstance(value, np.ndarray):
                    value = ndarray.array(value, ctx=self.ctx)
                    #node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
                    node_to_val_map[node] = value

                    ##assert isinstance(value, ndarray.NDArray) , "errrrooooorr !!!!"

                if isinstance(value, ndarray.NDArray):
                    if phase =="train":
                        value1 = ndarray.array(value.asnumpy(), ctx = ndarray.cpu(0))
                        value2 = ndarray.empty(shape = value.asnumpy().shape, ctx = ndarray.cpu(0))

                        if self.init_ps and self.init_count<len(self.eval_node_list):
                          if self.worker_id!='./W0':
                            # print(self.worker_id, " no", self.init_count)
                            zero_array=np.zeros(value.asnumpy().shape)
                            value1 = ndarray.array(zero_array, ctx = ndarray.cpu(0))
                            # print value1.asnumpy()[:]
                          else:
                            pass
                            # print(self.worker_id, " yes", self.init_count)
                          worker_communicate.DL_Communicate_BSP(node.id, value1.handle, value2.handle)
                          self.init_count+=1


                    node_to_val_map[node] = value
                
                if isinstance(value, scipy.sparse.spmatrix):
                    value = scipy.sparse.coo_matrix(value)
                    value = ndarray.sparse_array(value.data, 
                            (value.row, value.col), shape = value.shape, ctx=self.ctx)
                    node_to_val_map[node] = value
                
                
                if isinstance(value, ndarray.ND_Sparse_Array):
                    
                    #TODO: need to implement ps for sparse
                    node_to_val_map[node] = value
                    

                if not isinstance(value, (ndarray.NDArray, ndarray.ND_Sparse_Array)):
                    assert False, "feed_dict value type not supported"

        # print"xxxx"
        # collect shapes for all placeholders
        # infer shape if feed_shapes changed since last run
        # e.g. call run() on test data after trainng
        # print feed_shapes

        feed_shapes = {}
        for node in node_to_val_map:
            feed_shapes[node] = node_to_val_map[node].shape

        if(not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
            self.infer_shape(feed_shapes)
            self.feed_shapes = feed_shapes
            if (not use_numpy):
                self.memory_plan(self.feed_shapes)



        ### node_to_val_map preserve feed_value

        for node in self.topo_order:

            if node in node_to_val_map:
                continue

            input_vals = [node_to_val_map[n] for n in node.inputs]

            if use_numpy:
                node_val = np.empty(shape = self.node_to_shape_map[node])
            else:
                node_val = self.node_to_arr_map[node]

            #node.op.compute(node, input_vals, node_val, use_numpy)
            
            if node.id in ps_map:
              parameter_id = ps_map[node.id] ## grad node id to node id 
              node.op.compute(node,phase,parameter_id,input_vals, node_val, use_numpy) ## phase == "train" return new W, phase == "evaluate" return grad_W
            else:
             
              #print "hello"
              node.op.compute(node,input_vals, node_val, use_numpy)



            node_to_val_map[node] = node_val

        if not use_numpy and convert_to_numpy_ret_vals:
            return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]

        return [node_to_val_map[n] for n in self.eval_node_list]








    # def run(self, feed_dict, convert_to_numpy_ret_vals=False):
    #     """
    #     Parameters
    #     ----------
    #     feed_dict: a dictionary of node->np.ndarray supplied by user.
    #     convert_to_numpy_ret_vals: whether to convert ret vals to np.array

    #     Returns
    #     -------
    #     A list of values for nodes in eval_node_list. NDArray or np.ndarray.
    #     """
    #     def are_feed_shapes_equal(sa, sb):
    #         if (not isinstance(sa, dict)) or (not isinstance(sb, dict)):
    #             return False
    #         unmatched_item = set(sa.items()) ^ set(sb.items())
    #         return len(unmatched_item) == 0

    #     # Assume self.ctx is None implies numpy array and numpy ops.

    #     use_numpy = self.ctx is None
    #     node_to_val_map = {}
    #     for node, value in feed_dict.items():
    #       if self.policy == 'vdnn':
    #         assert isinstance(value, np.ndarray)
    #         node_to_val_map[node] = value
    #       else:
    #         if use_numpy:
    #             # all values passed in feed_dict must be np.ndarray
    #             assert isinstance(value, np.ndarray)
    #             node_to_val_map[node] = value
    #         else:
    #             # convert values to ndarray.NDArray if necessary
    #             if isinstance(value, np.ndarray):
    #                 if self.policy == 'swap':
    #                   if node.swap == True:
    #                     node_to_val_map[node] = value
    #                   else:
    #                     node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
    #                 else:
    #                   node_to_val_map[node] = ndarray.array(value, ctx=self.ctx)
    #             elif isinstance(value, ndarray.NDArray):
    #                 node_to_val_map[node] = value
    #             else:
    #                 assert False, "feed_dict value type not supported"

    #     # collect shapes for all placeholders
    #     feed_shapes = {}
    #     for node in node_to_val_map:
    #         feed_shapes[node] = node_to_val_map[node].shape

    #     # infer shape if feed_shapes changed since last run
    #     # e.g. call run() on test data after trainng
    #     # print feed_shapes
    #     if (not are_feed_shapes_equal(feed_shapes, self.feed_shapes)):
    #         self.infer_shape(feed_shapes)
    #         self.feed_shapes = feed_shapes
    #         if not self.policy == 'vdnn':
    #           # plan memory if using GPU
    #           if (not use_numpy):
    #             self.memory_plan(feed_shapes)
    #     # Traverse graph in topo order and compute values for all nodes.
    #     global FLAG_SHOW_GRAPH
    #     if self.policy == 'swap':
    #       # generate swap queue
    #       if not use_numpy:
    #         for node in self.topo_order:
    #           if node not in node_to_val_map:
    #             # variable in placeholder
    #             for input_node in node.inputs:
    #               if input_node.swap == True:
    #                 self.swap_queue.append(input_node)
    #             # variable grad
    #             if node.swap == True:
    #                 self.swap_queue.append(node)
    #         node_in_GPU = None
    #         if FLAG_SHOW_GRAPH:
    #           print "Show swap queue:"
    #           for node in self.swap_queue:
    #             print node
    #     elif self.policy == 'vdnn':
    #       # TODO traverse graph to select in-gpu window
    #       window = [0,0]
    #       if not use_numpy:
    #         nvmlInit()
    #         handle = nvmlDeviceGetHandleByIndex(0)
    #         info = nvmlDeviceGetMemoryInfo(handle)
    #         gpu_mem = info.free
    #         nvmlShutdown()
    #         loss_node = self.eval_node_list[0]
    #         window[1] = self.topo_order.index(loss_node)+1
    #         window[0] = self.topo_order.index(loss_node)+1
    #         for node in reversed(self.topo_order[:window[1]+1]):
    #           node_size = 4  # float32
    #           #print node, self.node_to_shape_map[node]
    #           for shape in self.node_to_shape_map[node]:
    #             node_size = node_size * shape
    #           if gpu_mem > node_size:
    #             gpu_mem = gpu_mem - node_size
    #             window[0] = window[0] - 1
    #         #print "gpu_mem:",gpu_mem
    #     # Traverse graph in topo order and compute values for all nodes. 
    #     if FLAG_SHOW_GRAPH:
    #       print "run topo_order"
    #     # Show graph dependency
    #     if FLAG_SHOW_GRAPH:
    #       print "node:",node
    #       print "node.desc:",node.desc

    #     for node in self.topo_order:
    #       if self.policy == 'vdnn':
    #         # Skip placeholder nodes
    #         if node in node_to_val_map:
    #           continue
    #         # H2D before compute
    #         ## Collect inputs 
    #         input_vals = []
    #         for n in node.inputs:
    #           if not use_numpy: 
    #             if isinstance(node_to_val_map[n], np.ndarray):
    #               node_to_val_map[n] = ndarray.array(node_to_val_map[n], ctx=self.ctx)
    #           input_vals.append(node_to_val_map[n])
    #         ## Alloc node space
    #         if use_numpy:
    #             node_val = np.empty(shape=self.node_to_shape_map[node])
    #         else:
    #             node_val = ndarray.empty(shape=self.node_to_shape_map[node], ctx=self.ctx)
    #         # Compute
    #         # node_val is modified in-place whether np.ndarray or NDArray
    #         node.op.compute(node, input_vals, node_val, use_numpy)
    #         # D2H after compute
    #         if use_numpy:
    #           node_to_val_map[node] = node_val
    #         else:
    #           node_index = self.topo_order.index(node)
    #           if node_index > window[0] and node_index < window[1]:
    #             node_to_val_map[node] = node_val
    #             continue
    #           node_to_val_map[node] = node_val.asnumpy()
    #           del node_val
    #           for n in node.inputs:
    #             if isinstance(node_to_val_map[n], ndarray.NDArray):
    #               tmp_val = node_to_val_map[n].asnumpy()
    #               del node_to_val_map[n]
    #               node_to_val_map[n] = tmp_val
    #       elif self.policy == 'swap':
    #         # Switch in GPU
    #         if not use_numpy:
    #           if self.swap_queue and (node_in_GPU==None):
    #             swap_node = self.swap_queue[0]
    #             if swap_node in node_to_val_map:
    #               node_to_val_map[swap_node] = ndarray.array(node_to_val_map[swap_node], ctx=self.ctx)
    #             else:
    #               self.node_to_arr_map[swap_node] = ndarray.empty(self.node_to_shape_map[swap_node], ctx=self.ctx) 
    #             node_in_GPU = swap_node.id
            
    #         if node in node_to_val_map:
    #             # Skip placeholder nodes. Values already provided by feed_dict.
    #             continue
    #         # Compute
    #         input_vals = [node_to_val_map[n] for n in node.inputs]
    #         if use_numpy:
    #             node_val = np.empty(shape=self.node_to_shape_map[node])
    #         else:
    #             node_val = self.node_to_arr_map[node]
    #         # node_val is modified in-place whether np.ndarray or NDArray
    #         node.op.compute(node, input_vals, node_val, use_numpy)
    #         if node.swap == True:
    #           node_to_val_map[node] = node_val.asnumpy()
    #           del node_val
    #           del self.node_to_arr_map[node]
    #           del self.swap_queue[0]
    #           node_in_GPU = None
    #         else:
    #           node_to_val_map[node] = node_val
    #           # Switch out GPU
    #           if not use_numpy:
    #             if self.swap_queue:
    #               if self.swap_queue[0] in node.inputs:
    #                 out_node = self.swap_queue.pop(0)
    #                 if self.swap_queue:
    #                   if not self.swap_queue[0].id == node_in_GPU:
    #                     tmp_array = node_to_val_map[out_node].asnumpy()
    #                     del node_to_val_map[out_node]
    #                     node_to_val_map[out_node] = tmp_array
    #                     node_in_GPU = None
    #       else:
    #         if node in node_to_val_map:
    #             # Skip placeholder nodes. Values already provided by feed_dict.
    #             continue
    #         input_vals = [node_to_val_map[n] for n in node.inputs]
    #         # print self.node_to_shape_map[node]

    #         if use_numpy:
    #             node_val = np.empty(shape=self.node_to_shape_map[node])
    #         else:
    #             node_val = self.node_to_arr_map[node]
    #         # node_val is modified in-place whether np.ndarray or NDArray
    #         # if (len(node.inputs) == 1):
    #         #     print "computs",node.inputs[0].name
    #         # else:
    #         #     print "computs",node.inputs[0].name,node.inputs[1].name
    #         # print node.name
                                
    #         # print node_val.shape
    #         # print "xxx"
    #         # print node.name
    #         node.op.compute(node, input_vals, node_val, use_numpy)
    #         # print "xxx"
    #         node_to_val_map[node] = node_val
    #         # print "xxx"

    #     if FLAG_SHOW_GRAPH:
    #       FLAG_SHOW_GRAPH = False
    #     # Collect node values.
    #     if not use_numpy and convert_to_numpy_ret_vals:
    #         if self.policy == 'swap':
    #           node_values = []
    #           for n in self.eval_node_list:
    #             if n.swap == True:
    #               node_values.append(node_to_val_map[n])
    #             else:
    #               node_values.append(node_to_val_map[n].asnumpy())
    #           return node_values
    #         elif self.policy == 'vdnn':
    #           return [node_to_val_map[n] for n in self.eval_node_list]
    #         else:             
    #           return [node_to_val_map[n].asnumpy() for n in self.eval_node_list]  
    #     return [node_to_val_map[n] for n in self.eval_node_list]


def ps_gradients(output_node, node_list, scheduler_policy=None):

    """Take gradient of output node with respect to each node in node_list.

    Parameters
    ----------
    output_node: output node that we are taking derivative of.
    node_list: list of nodes that we are taking derivative wrt.

    Returns
    -------
    A list of gradient values, one for each node in node_list respectively.

    """
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
                node_to_output_grad[node].swap=True
    # grad_node_list = [node_to_output_grad[node] for node in node_list]

    grad_node_list = [ps_communicate_op(node_to_output_grad[node], learning_rate=-1.0e-3/int(os.environ['DMLC_NUM_WORKER'])) for node in node_list]
    
    print("-------ps_gradient id to node id -------------")
    #print "len node_list",len(grad_node_list)
    
    ps_map ={}
    
    for i in range(len(node_list)):

      ps_map[grad_node_list[i].id]=node_list[i].id

      #print node_list[i].name, node_list[i].id, grad_node_list[i].id

    print('----------------END------------------------')
    return grad_node_list,ps_map








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
            # print node.name
            node_to_output_grads_list[node.inputs[i]].append(
                input_grads_list[i])
    if scheduler_policy == 'swap':
      for node in node_list:
        if node.swap:
          node_to_output_grad[node].swap=True
    grad_node_list = [node_to_output_grad[node] for node in node_list]
    # grad_node_list = [distributed_communicate_op(node_to_output_grad[node]) for node in node_list]
    return grad_node_list







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
          node_to_output_grad[node].swap=True
    # grad_node_list = [node_to_output_grad[node] for node in node_list]
    grad_node_list = [distributed_communicate_op(node_to_output_grad[node]) for node in node_list]
    return grad_node_list




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
    
