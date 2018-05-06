import numpy as np
import tensorflow as tf

import re

import audemo_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 250, 'Number of images to process')

tf.app.flags.DEFINE_boolean('use_fp16', False, 'Train the model using fp16.')

MOVING_AVERAGE_DECAY = 0.9999
NUM_EPOCHS_PER_DECAY = 350.0
LEARNING_RATE_DECAY_FACTOR = 0.8
INITIAL_LEARNING_RATE = 0.1

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = audemo_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = audemo_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

RNN_LAYERS = 2
RNN_HIDDEN_SIZE = 6
RNN_REDUCED_SIZE = 288
NUM_CLASSES = 4

IMAGE_HEIGHT = audemo_input.IMAGE_HEIGHT
IMAGE_WIDTH = audemo_input.IMAGE_WIDTH

TOWER_NAME = 'tower'


def train_input(num_epochs, is_augment=True):
    images, labels =  audemo_input.inputs(is_train=True, 
      batch_size=FLAGS.batch_size, 
      num_epochs=num_epochs, 
      is_augment=is_augment)
    return images, labels


def eval_input(num_epochs):
    images, labels =  audemo_input.inputs(is_train=False, 
      batch_size=FLAGS.batch_size, 
      num_epochs=num_epochs, 
      is_augment=False)
    return images, labels

def _activation_summary(x):
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_on_cpu(name, shape, initializer):
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(name,
                           shape,
                           tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))

    if wd is not None:
        weight_dacay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_dacay)
    return var


def inference(inputs, keep_prob):
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[7, 7, audemo_input.IMAGE_DEPTH, 16],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [16], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(value=conv, bias=biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    pool1 = tf.nn.max_pool(conv1,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')
    norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[5, 5, 16, 32],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(value=conv, bias=biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv2)

    norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    pool2 = tf.nn.max_pool(norm2,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 32],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(value=conv, bias=biases)
        conv3 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv3)

    pool3 = tf.nn.max_pool(conv3,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')
    norm3 = tf.nn.lrn(pool3, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights',
                                             shape=[3, 3, 32, 32],
                                             stddev=5e-2,
                                             wd=None)
        conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [32], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(value=conv, bias=biases)
        conv4 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv4)

    pool4 = tf.nn.max_pool(conv4,
                           ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')

    # return tf.shape(pool4)


    pool4 = tf.reshape(pool4, [-1, RNN_REDUCED_SIZE, 32])
    pool4_concat = [pool4]
    pool4_concat = tf.concat(pool4_concat, 2)
    inputs4 = [tf.squeeze(input_, [1]) for input_ in tf.split(pool4_concat, num_or_size_splits=RNN_REDUCED_SIZE, axis=1)]


    with tf.variable_scope('rnn5') as scope:
        cell = tf.contrib.rnn.GRUCell(128)
        init_state_cell = cell.zero_state(FLAGS.batch_size, tf.float32)
        output, final_state = tf.nn.static_rnn(
            cell, inputs4, initial_state=init_state_cell)


    with tf.variable_scope('flc6') as scope:
        reshape = tf.reshape(final_state, [FLAGS.batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights',
                                              shape=[dim, 176],
                                              stddev=0.04,
                                              wd=0.004)
        biases = _variable_on_cpu('biases', [176], tf.constant_initializer(0.1))

        # Dropout layer
        reshape = tf.nn.dropout(reshape, keep_prob=keep_prob)
        flc6 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        _activation_summary(flc6)

    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights',
                                              [176, NUM_CLASSES],
                                              stddev=1 / 176.0, wd=None)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(flc6, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear


def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def _add_loss_summaries(total_loss):
    loss_averages_calculator = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_group = tf.get_collection('losses')
    loss_averages_op = loss_averages_calculator.apply(loss_group + [total_loss])

    for loss in loss_group + [total_loss]:
        tf.summary.scalar(loss.op.name + '(raw)', loss)
        tf.summary.scalar(loss.op.name, loss_averages_calculator.average(loss))

    return loss_averages_op


def train(total_loss, global_step):
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)
    tf.summary.scalar('learning_rate', lr)

    loss_averages_op = _add_loss_summaries(total_loss)

    with tf.control_dependencies([loss_averages_op]):
        optimizer = tf.train.AdagradOptimizer(lr)
        grads = optimizer.compute_gradients(total_loss)

    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    variable_averages_calculator = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    with tf.control_dependencies([apply_gradient_op]):
        variables_averages_op = variable_averages_calculator.apply(tf.trainable_variables())

    return variables_averages_op
