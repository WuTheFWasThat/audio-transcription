from __future__ import absolute_import, division, print_function

import tensorflow as tf

import constants

class Model(object):
    def get_ops(self, X, Y):
        raise NotImplementedError

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


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
        w = init_weights([constants.NUM_FREQUENCIES, constants.NUM_NOTES])
        variable_summaries('weights.summaries', w)
        py_x = tf.matmul(X, w)

        # compute mean cross entropy (softmax is applied internally)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
        tf.summary.scalar('cross_entropy', cost)

        # construct optimizer
        train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
        # at predict time, evaluate the argmax of the logistic regression
        predict_op = tf.argmax(py_x, 1)
        return train_op, predict_op
