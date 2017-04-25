from __future__ import absolute_import, division, print_function

import os
import argparse
import time
import pickle
import logging
import sh

import tensorflow as tf
import numpy as np

import midi
import models
import constants
import utils

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('sh').setLevel(logging.WARN)
logging.basicConfig()

parser = argparse.ArgumentParser(description='Run a simple single-layer logistic model.')
parser.add_argument('--batch_size', '-b', default=32, type=int,
                    help='How many examples per batch')
parser.add_argument('--num_iters', '-n', default=0, type=int,
                    help='How many iterations to run, defaults to 0 meaning infinite')
parser.add_argument('--save_iters', default=100, type=int,
                    help='Frequency with which to save and log')
parser.add_argument('--name', default='model', type=str,
                    help='Name of run (determines locations to save it)')
parser.add_argument('--checkpoint', default=None, type=int,
                    help='Number of checkpoint to restore')
args = parser.parse_args()

checkpoints_dir = os.path.join(os.getcwd(), 'checkpoints', args.name)
sh.mkdir('-p', checkpoints_dir)
summaries_dir = os.path.join(os.getcwd(), 'summaries', args.name)
# remove old summaries so tensorboard doesn't pick them up
sh.rm('-rf', summaries_dir)
sh.mkdir('-p', summaries_dir)

# model = models.BasicLogistic()
model = models.BasicSequential()

if model.sequential():
    # first dimension batch size
    # second dimension time
    X = tf.placeholder("float", [None, None, constants.NUM_FREQUENCIES])
    Y = tf.placeholder("float", [None, constants.NUM_NOTES])
else:
    # first dimension batch size
    X = tf.placeholder("float", [None, constants.NUM_FREQUENCIES])
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
        if args.num_iters > 0 and step > args.num_iters:
            break

        spectrums = []
        answers = []
        labels = []

        for _ in xrange(args.batch_size):
            if model.sequential():
                data = midi.sampleLabeledSequentialData()
                spectrum = data['spectrums']
            else:
                data = midi.sampleLabeledData()
                spectrum = data['spectrum']
            answer = data['note']
            spectrums.append(
                (spectrum - np.mean(spectrum)) / np.std(spectrum)
            )
            answers.append(answer)
            labels.append(utils.one_hot(constants.NUM_NOTES, answer))

        # pad spectrums
        if model.sequential():
            max_len = max(
                np.shape(spectrum)[1] for spectrum in spectrums
            )
            spectrums = [
                np.pad(
                    spectrum,
                    mode='constant', constant_values=0,
                    pad_width=((0, 0), (0, max_len - np.shape(spectrum)[1]), (0, 0)),
                )
                for spectrum in spectrums
            ]

        summaries, _, predicted = sess.run(
            [summaries_op, train_op, predict_op],
            feed_dict={
                X: np.concatenate(spectrums, axis=0),
                Y: np.concatenate(labels, axis=0),
            }
        )

        ncorrect += np.sum(np.equal(answers, predicted))

        if step % args.save_iters == 0:
            logging.info('Iteration %d', step)

            # Save the variables to disk.
            save_path = saver.save(sess, os.path.join(checkpoints_dir, "checkpoint.%d" % step))
            logging.info("Model saved in file: %s", save_path)

            train_writer.add_summary(summaries, step)
            train_writer.add_summary(
                utils.make_summary('steps/sec', step/(time.time() - t)), step)

            # train_writer.add_summary(
            #     utils.make_summary('percent correct', (ncorrect + 0.0) / step), step)
            train_writer.add_summary(
                utils.make_summary(
                    'percent correct',
                    (ncorrect + 0.0) / (args.save_iters * args.batch_size)),
                step)
            ncorrect = 0
