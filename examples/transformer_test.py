from athena import gpu_ops as ad
import numpy as np
from athena import ndarray
from athena.microopOptimizer import microopOptimizer
from athena.microopPlanner import microopPlanner
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', default=32000, type=int)
parser.add_argument('--batch_size', default=16, type=int)

parser.add_argument('--d_model', default=512, type=int,
                        help="hidden dimension of encoder/decoder")
parser.add_argument('--d_ff', default=2048, type=int,
                    help="hidden dimension of feedforward layer")
parser.add_argument('--num_blocks', default=6, type=int,
                    help="number of encoder/decoder blocks")
parser.add_argument('--num_heads', default=8, type=int,
                    help="number of attention heads")
parser.add_argument('--maxlen1', default=100, type=int,
                    help="maximum length of a source sequence")
parser.add_argument('--maxlen2', default=100, type=int,
                    help="maximum length of a target sequence")
parser.add_argument('--dropout_rate', default=0.3, type=float)
parser.add_argument('--smoothing', default=0.1, type=float,
                    help="label smoothing rate")

parser.add_argument('-p','--policy', default='None')

executor_ctx = ndarray.gpu(0)
variable_list = []
val_list = []


rand = np.random.RandomState(seed=123)
def get_variable(name, size):
    global variable_list, val_list
    x = ad.Variable(name=name)
    x_val = rand.normal(scale=0.1, size=size)
    x_val = ndarray.array(x_val, ctx=executor_ctx)
    variable_list.append(x)
    val_list.append(x_val)
    return x

# def layer_norm(
#     input_tensor, 
#     feature_size, 
#     eps=1e-8
# ):
#     scale = init.ones(name='layer_norm_scale', shape=(feature_size, ))
#     bias = init.zeros(name='layer_norm_biad', shape=(feature_size, ))
#     return ad.batch_normalization_op(conv, bn_scale, bn_bias)


def dense(
    input_tensor, 
    fan_in, 
    fan_out, 
    activation=None, 
):
    weights = get_variable(name='dense_weight', size=(fan_in, fan_out))
    bias = get_variable(name="dense_bias", size=(fan_out,))
    outputs = ad.matmul_op(input_tensor, weights)
    outputs = outputs + ad.broadcasttoTF_op(bias, outputs)
    if activation is not None:
        outputs = activation(outputs)
    return outputs


def dropout(
    input_tensor, 
    dropout_prob
):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = ad.dropout_op(input_tensor, 1.0 - dropout_prob)
    return output


def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    if zero_pad:
        # embedding_part = initializer(name='embedding_table', shape=(vocab_size-1, num_units))
        # padding_zero = init.zeros(name='padding_zero', shape=(1, num_units))
        embedding_part = get_variable(name='embedding_table', size=(vocab_size-1, num_units))
        padding_zero = get_variable(name='padding_zero', size=(1, num_units))
        embeddings = ad.concat_op(padding_zero, embedding_part)
    else:
        # embeddings = initializer(name='embedding_table', shape=(vocab_size, num_units))
        embeddings = get_variable(name='embedding_table', size=(vocab_size, num_units))
    return embeddings


def multihead_attention(
    queries, keys, values,
    config,
    query_act=None, key_act=None, value_act=None,
    attention_mask=None,
    causality=False):

    def transpose_for_scores(input_tensor):
        output_tensor = ad.array_reshape_op(
            input_tensor, [config.batch_size, -1, config.num_heads, config.d_model // config.num_heads])

        output_tensor = ad.transpose_op(output_tensor, [0, 2, 1, 3])
        return output_tensor
    
    batch_size = config.batch_size
    hidden_size = config.d_model
    num_attention_heads = config.num_heads
    caus_len = config.maxlen2 - 1
    # attention_probs_dropout_prob = config.dropout_rate
    
    size_per_head = hidden_size // num_attention_heads

    # reshape to 2d
    queries2d = ad.array_reshape_op(queries, [-1, hidden_size]) # (N * T_q, d_model)
    keys2d = ad.array_reshape_op(keys, [-1, hidden_size]) # (N * T_k, d_model)
    values2d = ad.array_reshape_op(values, [-1, hidden_size]) # (N * T_k, d_model)

    # linear transformation
    query_layer = dense(queries2d, hidden_size, hidden_size, query_act) # (N * T_k, d_model)
    key_layer = dense(keys2d, hidden_size, hidden_size, key_act) # (N * T_k, d_model)
    value_layer = dense(values2d, hidden_size, hidden_size, value_act) # (N * T_k, d_model)

    # transpose
    query_layer = transpose_for_scores(query_layer) # (N, h, T_q, d_model/h)
    key_layer = transpose_for_scores(key_layer) # (N, h, T_k, d_model/h)
    value_layer = transpose_for_scores(value_layer) # (N, h, T_k, d_model/h)

    # score
    attention_scores = ad.batch_matmul_op(query_layer, key_layer, trans_B=True) # (N, h, T_q, T_k)
    attention_scores = attention_scores * (1.0 / np.sqrt(float(size_per_head)))

    # mask
    if attention_mask is not None:
        # zeros = ad.Variable('no_mask', value=np.array((0,), dtype=np.float32), trainable=False)
        zeros = get_variable(name = 'no_mask', size=(1,))
        # adder = ad.Variable('attention_mask', value=np.array((-2**32+1,), dtype=np.float32), trainable=False)
        adder = get_variable(name = 'attention_mask', size=(1,))
        zeros = ad.broadcasttoTF_op(zeros, attention_mask)
        adder = ad.broadcasttoTF_op(adder, attention_mask)
        attention_mask = ad.where_op(attention_mask, zeros, adder) # (N, T)
        attention_mask = ad.array_reshape_op(attention_mask, [batch_size, 1, 1, -1])
        attention_scores = attention_scores + ad.broadcasttoTF_op(attention_mask, attention_scores)
    if causality:
        # tril = ad.Variable(name='tril', value=np.tril(np.ones((caus_len, caus_len))), trainable=False) # (T, T)
        tril = get_variable(name='tril', size=(caus_len, caus_len)) # (T, T)
        future_masks = ad.broadcast_shape_op(tril, [batch_size, num_attention_heads, caus_len, caus_len])
        # adder = ad.Variable('future_mask', value=np.array((-2**32+1,), dtype=np.float32), trainable=False)
        adder = get_variable(name='future_mask', size=(1,))
        adder = ad.broadcasttoTF_op(adder, future_masks)        
        attention_scores = ad.where_op(future_masks, attention_scores, adder) # (N, h, T, T)
    
    # probs
    attention_probs = attention_scores
    # attention_probs = ad.softmax_op(attention_scores)
    # attention_probs = dropout(attention_probs, attention_probs_dropout_prob)
    context_layer = ad.batch_matmul_op(attention_probs, value_layer)
    context_layer = ad.transpose_op(context_layer, [0, 2, 1, 3])
    outputs = ad.array_reshape_op(
        context_layer,
        [batch_size, -1, num_attention_heads * size_per_head])
    
    # Residual connection
    outputs = outputs + queries  # (N, T_q, d_model)

    # Normalize
    # outputs = layer_norm(outputs, hidden_size)  # (N, T_q, d_model)
    return outputs


def ff(inputs, config):
    outputs = ad.array_reshape_op(inputs, [-1, config.d_model])
    outputs = dense(outputs, config.d_model, config.d_ff, activation=ad.relu_op)
    outputs = dense(outputs, config.d_ff, config.d_model)
    outputs = ad.array_reshape_op(outputs, [config.batch_size, -1, config.d_model])
    outputs = outputs + inputs
    # outputs = layer_norm(outputs, config.d_model)
    return outputs


def label_smoothing(inputs, V, epsilon=0.1):
    # V = inputs.shape[-1] # number of channels
    return ((1-epsilon) * inputs) + (epsilon / V)


def positional_encoding(
    inputs,
    inputs_shape,
    maxlen,
    masking=True
):
    N, T, E = tuple(inputs_shape)
    position_enc = np.array([
            [pos / np.power(10000, (i & -2)/E) for i in range(E)]
            for pos in range(maxlen)])
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    
    position_enc = position_enc[:T, :]
    shape = list(position_enc.shape)
    shape.insert(0, N)
    # outputs = ad.Variable(name='position_enc', value=np.tile(position_enc, [N, 1, 1]), trainable=False)
    outputs = get_variable(name='position_enc', size =shape)
    # zeros = ad.Variable(name='zeros', value=np.zeros(inputs_shape), trainable=False)
    zeros = get_variable(name='zeros', size = inputs_shape)

    if masking:
        outputs = ad.where_op(inputs, outputs, zeros)

    return outputs


class Transformer(object):
    def __init__(self, hp):
        self.hp = hp
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=False)

    def encode(self, xs):
        x = xs

        # embedding
        enc = ad.embedding_lookup_op(self.embeddings, x) # (N, T1, d_model)
        enc = enc * self.hp.d_model**0.5 # scale

        enc += positional_encoding(enc, (self.hp.batch_size, self.hp.maxlen1, self.hp.d_model), self.hp.maxlen1)
        # enc = dropout(enc, self.hp.dropout_rate)

        ## Blocks
        for i in range(self.hp.num_blocks):
            # self-attention
            enc = multihead_attention(
                queries=enc, keys=enc, values=enc,
                config=self.hp,
                attention_mask=x,
                causality=False
            )
            # feed forward
            enc = ff(enc, config=self.hp)
        memory = enc
        return memory

    def decode(self, ys, memory, src_masks):
        decoder_inputs = ys

        # embedding
        dec = ad.embedding_lookup_op(self.embeddings, decoder_inputs)  # (N, T2, d_model)
        dec = dec * self.hp.d_model ** 0.5  # scale

        dec += positional_encoding(dec, (self.hp.batch_size, self.hp.maxlen2-1, self.hp.d_model), self.hp.maxlen2)
        # dec = dropout(dec, self.hp.dropout_rate)

        # Blocks
        for i in range(self.hp.num_blocks):
            # Masked self-attention (Note that causality is True at this time)
            dec = multihead_attention(
                queries=dec, keys=dec, values=dec,
                config=self.hp,
                attention_mask=decoder_inputs,
                causality=True,
            )
            # Vanilla attention
            dec = multihead_attention(
                queries=dec, keys=memory, values=memory,
                config=self.hp,
                attention_mask=src_masks,
                causality=False,
            )
            ### Feed Forward
            dec = ff(dec, config=self.hp)

        dec = ad.array_reshape_op(dec, [-1, self.hp.d_model]) # (N * T, d_model)
        logits = ad.array_reshape_op(ad.matmul_op(dec, self.embeddings, trans_B=True), [self.hp.batch_size, -1, self.hp.vocab_size]) # (N, T, vocab)

        return logits

    def train(self, xs, ys):
        # forward
        memory = self.encode(xs)
        logits = self.decode(ys[0], memory, xs)
        
        # train scheme
        y = ys[1]
        y_ = label_smoothing(ad.one_hot_op(y, self.hp.vocab_size), self.hp.vocab_size) # (N, T, vocab)
        loss = ad.softmaxcrossentropy_op(logits, y_)
        return loss

def transformer(batch_size, policy = "None"):
    global variable_list, val_list
    variable_list = []
    val_list = []
    hp = parser.parse_args()
    hp.batch_size = batch_size
    xs = ad.Variable(name='X')
    X_val = np.empty(shape=(hp.batch_size, 100), dtype=np.float32)
    # X_val = ndarray.array(X_val, ctx=executor_ctx)
    ys1 = ad.Variable(name='y_')
    ys1_val = np.empty(shape=(hp.batch_size, 99), dtype=np.float32)
    # ys1_val = ndarray.array(ys1_val, ctx=executor_ctx)

    ys2 = ad.Variable(name='y_')
    ys2_val = np.empty(shape=(hp.batch_size, 99), dtype=np.float32)
    # ys2_val = ndarray.array(ys2_val, ctx=executor_ctx)

    nonpadding = ad.Variable(name='nonpadding')
    m = Transformer(hp)
    loss = m.train(xs, (ys1, ys2))
    # loss = ad.div_op(ad.reduce_sum_op(loss * nonpadding, axes=[0, 1]), ad.reduce_sum_op(nonpadding, axes=[0, 1]) + 1e-7)
    
    grad_list = ad.gradients(loss, variable_list)

    if policy == "None" or policy == "base":
        athena_exec = ad.Executor
    elif policy == "vdnnconv" or policy == "vdnnall":
        athena_exec = ad.vdnnExecutor
    elif policy == "superneurons":
        athena_exec = ad.superNeuronsExecutor
    elif policy == "recompute_memory" or policy == "recompute_speed":
        athena_exec = ad.recomputeExecutor
    elif policy == "simulator":
        athena_exec = microopOptimizer
    elif policy == "profiler":
        athena_exec = ad.profileExecutor
    elif policy == "planner":
        athena_exec = microopPlanner
    elif policy == "tsplit":
        athena_exec = ad.microopExecutor
    else:
        raise NotImplementedError

    if policy == "vdnnconv":
        executor = athena_exec([loss] + grad_list + [ys1, ys2], ctx=executor_ctx, policy = "conv")
    elif policy == "vdnnall": 
        executor = athena_exec([loss] + grad_list +  [ys1, ys2], ctx=executor_ctx, policy = "all")
    elif policy == "recompute_memory":
        executor = athena_exec([loss] + grad_list +  [ys1, ys2], ctx=executor_ctx, policy = "memory")
    elif policy == "recompute_speed":
        executor = athena_exec([loss] + grad_list +  [ys1, ys2], ctx=executor_ctx, policy = "speed")
    else:
        executor = athena_exec([loss] + grad_list +  [ys1, ys2], ctx=executor_ctx)

    feed_dict = dict()
    feed_dict[xs] = X_val
    feed_dict[ys1] = ys1_val
    feed_dict[ys2] = ys2_val
    for i in range(len(variable_list)):
        feed_dict[variable_list[i]] = val_list[i]
    import time
    print("running begin")
    for i in range(2):
        print("epoch:", i)
        if i == 1:
            start = time.time()
        grad_val_list = executor.run(feed_dict)

    end = time.time()
    return (end - start) / 1

if __name__ == "__main__":
    print("Building Transformer!")
    args = parser.parse_args()
    policy = args.policy

    batch_size = 256
    print("batch size = ", batch_size)
    execution_time = transformer(batch_size, policy = "tsplit")
    print("execution time = ", execution_time)

    # output_file_name = "/home/xiaonan/microop/Athena/exp/" + "transformer" + "/" + policy + "_batchsize_with_time.txt"
    # output_file = open(output_file_name, "a+", buffering=1)
    # output_file.write("Policy: {}, on Transformer\n".format(policy))

    # for batch_size in range(1, 1000, 1):
    #     execution_time = transformer(batch_size, policy = policy)
    #     print("Batch size: {} , time: {} s\n".format(batch_size, execution_time))
    #     output_file.write("Batch size: {} , time: {} s\n".format(batch_size, execution_time))
    # output_file.close()