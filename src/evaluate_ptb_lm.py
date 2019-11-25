import sys
import numpy as np
is_training = True
Py3 = sys.version_info[0] == 3
CUDNN = "cudnn"
num_steps = 35
def _get_lstm_cell():
    return tf.contrib.rnn.LSTMBlockCell(hidden_size, forget_bias=0.0)

def make_cell():
    cell = _get_lstm_cell()
    return cell

batch_size = 20

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
  data = _read_words(filename)

  counter = collections.Counter(data)
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

  words, _ = list(zip(*count_pairs))
  word_to_id = dict(zip(words, range(len(words))))

  return word_to_id

def _file_to_word_ids(filename, word_to_id):
  data = _read_words(filename)
  return [word_to_id[word] for word in data if word in word_to_id]

def ptb_raw_data(data_path=None):
  """Load PTB raw data from data directory "data_path".
  Reads PTB text files, converts strings to integer ids,
  and performs mini-batching of the inputs.
  The PTB dataset comes from Tomas Mikolov's webpage:
  http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.
  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """

  train_path = os.path.join(data_path, "train.txt")
  valid_path = os.path.join(data_path, "valid.txt")
  test_path = os.path.join(data_path, "test.txt")

  word_to_id = _build_vocab(train_path)
  train_data = _file_to_word_ids(train_path, word_to_id)
  valid_data = _file_to_word_ids(valid_path, word_to_id)
  test_data = _file_to_word_ids(test_path, word_to_id)
  vocabulary = len(word_to_id)
  return train_data, valid_data, test_data, vocabulary
def str2bool(arg):
    return arg == 'True'

def data_type():
  return tf.float32 if 1 else tf.float32

import tensorflow as tf

import os
import collections
import argparse

parser = argparse.ArgumentParser(description='Build Language Model')
parser.add_argument('--data', type=str, help='data dir to ptb data set')
parser.add_argument('--large', type=str2bool, help='ptb-large or ptb-small')
parser.add_argument('--W1', type=str, default='',help='data dir to W1')
parser.add_argument('--W2', type=str, default='',help='data dir to W2')
args = parser.parse_args()
if args.large:
    hidden_size = 1500
else:
    hidden_size = 200
    


def __main__():
    
    
    raw_data = ptb_raw_data(args.data + "/simple-examples/")
    train_data, valid_data, test_data, _ = raw_data

    _input = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="input")
    _targets = tf.placeholder(tf.int32, shape=(batch_size, num_steps), name="targets")
    vocab_size = 10000
    keep_prob = 0.35
    num_layers = 2

    init_scale = 0.04
    initializer = tf.random_uniform_initializer(-init_scale,init_scale)
    max_epoch = 14
    lr_decay = 1/1.15
    lrlr = 1.



    with tf.variable_scope('MODEL', initializer=initializer,reuse=False):
        learning_rate = tf.Variable(1., trainable=False)
        embedding = tf.get_variable("embedding", [vocab_size, hidden_size], dtype=data_type())
        inputs = tf.nn.embedding_lookup(embedding, _input)
        val_inputs = inputs
        inputs = tf.nn.dropout(inputs, keep_prob)
        val_cell = [make_cell() for _ in range(num_layers)]
        cell = [tf.contrib.rnn.DropoutWrapper(c,output_keep_prob=keep_prob) for c in val_cell]
        the_cell = tf.contrib.rnn.MultiRNNCell(cell, state_is_tuple=True)
        #val_cell = tf.contrib.rnn.MultiRNNCell(val_cell)
        initial_state = the_cell.zero_state(batch_size, data_type())
        #inputs = tf.unstack(inputs, num=num_steps, axis=1)
        #outputs, final_state = tf.nn.static_rnn(the_cell, inputs,initial_state=initial_state)
        #cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

        outputs, final_state = tf.nn.dynamic_rnn(the_cell, inputs, initial_state=initial_state,dtype=data_type())
        #val_outputs, val_final_state = tf.nn.dynamic_rnn(val_cell, val_inputs, initial_state=initial_state,dtype=tf.float32)
        #val_output = tf.reshape(tf.concat(val_outputs, 1), [-1, hidden_size])
        output = tf.reshape(tf.concat(outputs, 1), [-1, hidden_size])
        softmax_w = tf.get_variable("softmax_W", [hidden_size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_B", [vocab_size], dtype=data_type())
        logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
        logits = tf.reshape(logits, [batch_size, num_steps, vocab_size])
        #val_logits = tf.nn.xw_plus_b(val_output, softmax_w, softmax_b)
        #val_logits = tf.reshape(val_logits, [batch_size, num_steps, vocab_size])

        loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                _targets,
                tf.ones([batch_size, num_steps], dtype=data_type()),
                average_across_timesteps=False,
                average_across_batch=True)
        cost = tf.reduce_sum(loss)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),10)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars))

        new_lr = tf.placeholder(tf.float32, shape=[])
        lr_update = tf.assign(learning_rate, new_lr)

    with tf.variable_scope('MODEL', initializer=initializer,reuse=True):
        val_cell = tf.contrib.rnn.MultiRNNCell(val_cell)
        val_outputs, val_final_state = tf.nn.dynamic_rnn(val_cell, val_inputs, initial_state=initial_state,dtype=tf.float32)
        val_output = tf.reshape(tf.concat(val_outputs, 1), [-1, hidden_size])
        val_logits = tf.nn.xw_plus_b(val_output, softmax_w, softmax_b)
        val_logits = tf.reshape(val_logits, [batch_size, num_steps, vocab_size])

        val_loss = tf.contrib.seq2seq.sequence_loss(
                val_logits,
                _targets,
                tf.ones([batch_size, num_steps], dtype=data_type()),
                average_across_timesteps=False,
                average_across_batch=True)
        val_cost = tf.reduce_sum(val_loss)

    config = tf.ConfigProto(
        gpu_options = tf.GPUOptions(allow_growth=True,visible_device_list="0"),
        device_count = {'GPU': 1},

    #    log_device_placement = True                                                                                     \

    )
    sess = tf.Session(config = config)
    sess.run(tf.global_variables_initializer())

    import numpy as np

    data_len = len(train_data)
    batch_len = data_len // num_steps
    num_batches = int(data_len / (batch_size * num_steps))
    xdata = train_data[:num_batches * batch_size * num_steps]
    ydata = np.copy(xdata)
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    x_batches = np.split(np.asarray(xdata).reshape(batch_size, -1), num_batches, 1)
    y_batches = np.split(np.asarray(ydata).reshape(batch_size, -1), num_batches, 1)
    valid_len = len(valid_data)
    num_batches = int(valid_len / (batch_size * num_steps))
    vdata = valid_data[:num_batches * batch_size * num_steps]
    vydata = np.copy(vdata)
    vydata[:-1] = vdata[1:]
    vydata[-1] = vdata[0]
    x_val_batches = np.split(np.asarray(vdata).reshape(batch_size, -1), num_batches, 1)
    y_val_batches = np.split(np.asarray(vydata).reshape(batch_size, -1), num_batches, 1)


    test_len = len(test_data)
    num_batches = int(test_len / (batch_size * num_steps))
    tdata = test_data[:num_batches * batch_size * num_steps]
    tydata = np.copy(tdata)
    tydata[:-1] = tdata[1:]
    tydata[-1] = tdata[0]
    x_test_batches = np.split(np.asarray(tdata).reshape(batch_size, -1), num_batches, 1)
    y_test_batches = np.split(np.asarray(tydata).reshape(batch_size, -1), num_batches, 1)
    if args.large:
        Vars = np.load(args.data + "/Large_LSTM.npy")
    else:
        Vars = np.load(args.data + "/Small_LSTM.npy")
        
    for k,v in enumerate(tvars):
        sess.run(tf.assign(v,Vars[k]))
        
    W1 = sess.run(tvars[0])
    W2 = sess.run(tvars[-2])
    W2 = W2.transpose()
    vocab_size = 10000
    a = np.zeros(vocab_size)
    for t in train_data:
        a[t] += 1
    a = np.asarray(a)
    
    if args.W1 !="":
        W1 =  np.load(args.W1)
    if args.W2 !="":
        W2 =  np.load(args.W2)
    
    
    sess.run(tf.assign(tvars[0],W1))
    sess.run(tf.assign(tvars[-2],W2.transpose()))
    # VALIDATE #
    stateb = sess.run(initial_state)
    t = np.arange(len(x_test_batches))
    cc = 0.0
    cnt = 0
    for k,idx in enumerate(t):
        feed_dict = {}
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = stateb[i].c
            feed_dict[h] = stateb[i].h
        feed_dict[_input] = x_test_batches[idx]
        feed_dict[_targets] = y_test_batches[idx]    
        _c,stateb = sess.run([val_cost,val_final_state],feed_dict=feed_dict)
        cc += _c
        cnt += num_steps
    perp = np.exp(cc/cnt)
    print("Test Ppl.: %.2f" % perp)
__main__()