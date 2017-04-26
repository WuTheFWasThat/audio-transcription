from __future__ import absolute_import, division, print_function

import tensorflow as tf

import constants

class Model(object):
    def get_ops(self, X, Y):
        raise NotImplementedError
    def sequential(self):
        raise NotImplementedError

def init_weights(shape, stddev=0.01, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=stddev), trainable=True, name=name)


def variable_summaries(scope, var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(scope):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

class BasicLogistic(Model):
    def get_ops(self, X, Y):
        w = init_weights([constants.NUM_FREQUENCIES, constants.NUM_NOTES], name='weights')
        variable_summaries('weights.summaries', w)
        p_y = tf.matmul(X, w)

        # compute mean cross entropy (softmax is applied internally)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p_y, labels=Y))
        tf.summary.scalar('cross_entropy', cost)

        # construct optimizer
        train_op = tf.train.AdamOptimizer().minimize(cost)
        # at predict time, evaluate the argmax of the logistic regression
        predict_op = tf.argmax(p_y, 1)
        return train_op, predict_op

    def sequential(self):
        return False

class BasicSequential(Model):
    def __init__(self):
         super(Model, self).__init__()

    def get_ops(self, X, Y):
        hidden_state_size = 50
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_state_size)
        # tf.contrib.rnn.GRUCell(hidden_state_size)

        output, _final_state = tf.nn.dynamic_rnn(
            lstm_cell, X, dtype=tf.float32
        )

        output = tf.transpose(output, [1, 0, 2])
        last = output[-1]

        w = init_weights([hidden_state_size, constants.NUM_NOTES], name='weights')
        b = tf.Variable(tf.constant(0.0, shape=[constants.NUM_NOTES]), trainable=True, name='biases')
        variable_summaries('weights.summaries', w)
        variable_summaries('bias.summaries', b)

        p_y = tf.matmul(last, w) + b

        # compute mean cross entropy (softmax is applied internally)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=p_y, labels=Y))
        tf.summary.scalar('cross_entropy', cost)

        grads = tf.gradients(cost, tf.trainable_variables())

        max_gradient_norm = 100
        clipped_grads, norm = tf.clip_by_global_norm(grads, max_gradient_norm)

        clipped_grads_and_vars = list(zip(clipped_grads, tf.trainable_variables()))
        grads_and_vars = list(zip(grads, tf.trainable_variables()))

        # Op to update all variables according to their gradient
        train_op = tf.train.AdamOptimizer().apply_gradients(
            grads_and_vars=clipped_grads_and_vars)

        for grad, var in grads_and_vars:
            tf.summary.histogram(var.name + '/gradient', grad)

        # construct optimizer
        # train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
        # at predict time, evaluate the argmax of the logistic regression
        predict_op = tf.argmax(p_y, 1)
        return train_op, predict_op

    def sequential(self):
        return True

