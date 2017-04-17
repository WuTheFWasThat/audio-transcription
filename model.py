from __future__ import absolute_import, division, print_function

import os
import argparse
import generate
import time

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np


def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name,
                                                                simple_value=val)])

parser = argparse.ArgumentParser(description='Run a simple single-layer logistic model.')
parser.add_argument('--niters', default=10000, type=int,
                    help='How many iterations to run')
parser.add_argument('--name', default='model', type=str,
                    help='Name of run (determines locations to save it)')
args = parser.parse_args()

checkpoints_dir = os.path.join(os.getcwd(), 'checkpoints', args.name)
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
summaries_dir = os.path.join(os.getcwd(), 'summaries', args.name)
if not os.path.exists(summaries_dir):
    os.makedirs(summaries_dir)
data_dir = os.path.join(os.getcwd(), 'data', args.name)
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    return tf.matmul(X, w)

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

num_frequencies = 2205
num_notes = 120

X = tf.placeholder("float", [None, num_frequencies]) # create symbolic variables
Y = tf.placeholder("float", [None, num_notes])

w = init_weights([num_frequencies, num_notes])
with tf.name_scope('weights'):
    variable_summaries(w)

py_x = model(X, w)

# compute mean cross entropy (softmax is applied internally)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
tf.summary.scalar('cross_entropy', cost)

# construct optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
# at predict time, evaluate the argmax of the logistic regression
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

summaries_op = tf.summary.merge_all()

training_data = [
    generate.sampleLabeledData()
    for _ in xrange(args.niters)
]

t = time.time()

# Launch the graph in a session
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

    # you need to initialize all variables
    tf.global_variables_initializer().run()

    # restore from checkpoint
    if False:
        saver.restore(sess, os.path.join(checkpoints_dir, "checkpoint.something"))

    for step in xrange(args.niters):
        if step % 100 == 0:
            print('Iteration %d' % step)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(checkpoints_dir, "checkpoint.%d" % step))
            print("Model saved in file: %s" % save_path)

            for _ in range(10):
                (frequencies, answer) = generate.sampleLabeledData()
                predicted = sess.run(predict_op, feed_dict={X: frequencies, Y: answer})[0]
                if np.argmax(answer) == predicted:
                    print('CORRECT!', np.argmax(answer))
                else:
                    print('INCORRECT!', np.argmax(answer), predicted)

        (frequencies, answer) = training_data[step]
        # train
        summary, _ = sess.run([summaries_op, train_op], feed_dict={X: frequencies, Y: answer})
        if step % 100 == 0:
            train_writer.add_summary(summary, step)
            train_writer.add_summary(make_summary('steps/sec', (time.time() - t)/(step + 1)), step)

    for i in range(10):
        (frequencies, answer) = generate.sampleLabeledData()
        predicted = sess.run(predict_op, feed_dict={X: frequencies, Y: answer})[0]
        if np.argmax(answer) == predicted:
            print('CORRECT!', np.argmax(answer))
        else:
            print('INCORRECT!', np.argmax(answer), predicted)

    # print(i, np.mean(np.argmax(teY, axis=1) ==
    #                  sess.run(predict_op, feed_dict={X: teX, Y: teY})))
