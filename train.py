from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import pickle
import logging
import sh

import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import numpy as np

import midi
import models
import constants

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('sh').setLevel(logging.WARN)
logging.basicConfig()

parser = argparse.ArgumentParser(description='Run a simple single-layer logistic model.')
parser.add_argument('--batch_size', '-b', default=1, type=int,
                    help='How many examples per batch')
parser.add_argument('--niters', '-n', default=0, type=int,
                    help='How many iterations to run, defaults to 0 meaning infinite')
parser.add_argument('--name', default='model', type=str,
                    help='Name of run (determines locations to save it)')
parser.add_argument('--checkpoint', default=None, type=int,
                    help='Number of checkpoint to restore')
args = parser.parse_args()

checkpoints_dir = os.path.join(os.getcwd(), 'checkpoints', args.name)
sh.mkdir('-p', checkpoints_dir)
summaries_dir = os.path.join(os.getcwd(), 'summaries', args.name)
sh.rm('-rf', summaries_dir)
sh.mkdir('-p', summaries_dir)

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name,
                                                                simple_value=val)])

def one_hot(n, i):
    assert i < n
    assert i >= 0
    return np.reshape([int(x == i) for x in xrange(n)], (1, -1))

model = models.BasicLogistic()

X = tf.placeholder("float", [None, constants.NUM_FREQUENCIES]) # create symbolic variables
Y = tf.placeholder("float", [None, constants.NUM_NOTES])

(train_op, predict_op) = model.get_ops(X, Y)

saver = tf.train.Saver(max_to_keep=1)

summaries_op = tf.summary.merge_all()

t = time.time()
logging.info('Beginning training')

# Launch the graph in a session
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

    # you need to initialize all variables
    tf.global_variables_initializer().run()

    # restore from checkpoint
    if args.checkpoint:
        saver.restore(sess, os.path.join(checkpoints_dir, "checkpoint.%d" % args.checkpoint))

    step = 0
    ncorrect = 0
    nexamples = 0
    while True:
        step += 1
        if args.niters > 0 and step > args.niters:
            break

        spectrums = []
        answers = []
        labels = []
        for _ in xrange(args.batch_size):
            (spectrum, answer) = midi.sampleLabeledData()
            spectrums.append(spectrum)
            answers.append(answer)
            labels.append(one_hot(constants.NUM_NOTES, answer))
        summaries, _, predicted = sess.run(
            [summaries_op, train_op, predict_op],
            feed_dict={
                X: np.concatenate(spectrums, axis=0),
                Y: np.concatenate(labels, axis=0),
            }
        )

        ncorrect += np.sum(np.equal(answers, predicted))

        if step % 100 == 0:
            logging.info('Iteration %d', step)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(checkpoints_dir, "checkpoint.%d" % step))
            logging.info("Model saved in file: %s", save_path)

            train_writer.add_summary(summaries, step)
            train_writer.add_summary(
                make_summary('steps/sec', step/(time.time() - t)), step)

            # train_writer.add_summary(
            #     make_summary('percent correct', (ncorrect + 0.0) / step), step)
            train_writer.add_summary(
                make_summary('percent correct',
                             (ncorrect + 0.0) / (100 * args.batch_size)), step)
            ncorrect = 0
