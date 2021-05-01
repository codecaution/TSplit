from __future__ import absolute_import
from .Node import Op, NAME_RULE, PROFILING_MODE
import numpy as np
from .. import profiler
from .._base import get_array_memory

class EmbeddingLookUp(Op):
    def __call__(self, embedding, index):
        new_node = Op.__call__(self)

        new_node.inputs = [embedding, index]
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()

        if NAME_RULE == 0:
            new_node.name = "(%s+%s)" % (embedding.name, index.name)
        elif NAME_RULE == 1:
            new_node.name = "EmbeddingLookUp"
        else:
            new_node.name = "EmbeddingLookUp"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s)" % (embedding.name, index.name)
        return new_node

    def profile(self, node, input_vals, output_val, is_static = True):
        if is_static:
            pass
        else:
            import time
            start = time.time()
            from ..gpu_links import embedding_lookup
            embedding_lookup(
                input_vals[0], input_vals[1], output_val, None)
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True ,stream_handle=None):
        from ..gpu_links import embedding_lookup

        assert len(input_vals) == 2
        if use_numpy:
            flatten_index = input_vals[1].asnumpy().reshape(-1).astype(np.int32)
            output_val[:] = input_vals[0].asnumpy()[flatten_index].reshape(output_val.shape)
        else:
            # pass
            embedding_lookup(
                input_vals[0], input_vals[1], output_val, stream_handle)

    def gradient(self, node, output_grad):
        return [embedding_lookup_gradient_op(output_grad, node.inputs[1], node.inputs[0]), None]

    def infer_shape(self, node, input_shapes):
        assert len(input_shapes) == 2
        output_shape = list(input_shapes[1])
        output_shape.append(input_shapes[0][1])
        return tuple(output_shape)

class EmbeddingLookUp_Gradient(Op):
    def __call__(self, vectors, index, embedding):
        new_node = Op.__call__(self)
        new_node.inputs = [vectors, index, embedding]
        if PROFILING_MODE == 1:
            new_node.profiler = profiler.CreateProfiler()

        if NAME_RULE == 0:
            new_node.name = "(%s+%s+%s)" % (vectors.name, embedding.name, index.name)
        elif NAME_RULE == 1:
            new_node.name = "EmbeddingLookUp_Gradient"
        else:
            new_node.name = "EmbeddingLookUp_Gradient"+str(new_node.id)
            new_node.desc = new_node.name + \
                "(%s, %s, %s)" % (vectors.name, embedding.name, index.name)
        return new_node

    def profile(self, node, input_vals, output_val, is_static = True):
        if is_static:
            pass
        else:
            import time
            start = time.time()
            pass
            node.profiler.time = (time.time() - start) * 1000

    def compute(self, node, input_vals, output_val, use_numpy=True ,stream_handle=None):
        assert len(input_vals) == 3
        from ..gpu_links import embedding_lookup_gradient
        embedding_lookup_gradient(input_vals[0], input_vals[1], output_val, stream_handle)


    def gradient(self, node, output_grad):
        raise NotImplementedError

    def infer_shape(self, node, input_shapes):
        return input_shapes[2]

def embedding_lookup_op(embedding, index):
    """Make a new instance of EmbeddingLookUp and call the instance.
    Parameters:
    ----
    embedding : Node
        The Node of Embedding.
    index : Node
        The index to be looked up.
    Returns:
    ----
    A new Node instance created by Op.
    """
    return EmbeddingLookUp()(embedding, index)

def embedding_lookup_gradient_op(vectors, index, embedding):
    """Make a new instance of EmbeddingLookUp_Gradient and call the instance.
    Parameters:
    ----
    vectors : Node
        Vectors which looked up from Embedding.
    index : Node
        The index to be looked up.
    Returns:
    ----
    A new Node instance created by Op.
    """    
    return EmbeddingLookUp_Gradient()(vectors, index, embedding)