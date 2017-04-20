"""
Find inputs on which the model is failing
"""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import time

import tensorflow as tf

import constants
import midi
import models
import utils

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('sh').setLevel(logging.WARN)
logging.basicConfig()

parser = argparse.ArgumentParser(description='Find a failing input for a given model.')
parser.add_argument('--name', default='model', type=str,
                    help='Name of run (determines locations to save it)')
parser.add_argument('--checkpoint', type=int,
                    help='Number of checkpoint to restore')
parser.add_argument('--max_iters', default=0, type=int,
                    help='Max number of iterations to search for input (default infinite)')
args = parser.parse_args()

checkpoints_dir = os.path.join(os.getcwd(), 'checkpoints', args.name)
if not args.checkpoint:
    logging.error('No checkpoint provided!  Please pass an integer to the --checkpoint flag')
    sys.exit(1)

model = models.BasicLogistic()

X = tf.placeholder("float", [None, constants.NUM_FREQUENCIES]) # create symbolic variables
Y = tf.placeholder("float", [None, constants.NUM_NOTES])

(train_op, predict_op) = model.get_ops(X, Y)

saver = tf.train.Saver()

t = time.time()
logging.info('Beginning training')

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.global_variables_initializer().run()

    # restore from checkpoint
    saver.restore(sess, os.path.join(checkpoints_dir, "checkpoint.%d" % args.checkpoint))

    step = 0
    while True:
        step += 1
        if args.max_iters > 0 and step > args.max_iters:
            logging.info('Failed to find incorrect prediction')
            break

        data = midi.sampleLabeledData()
        spectrum = data['spectrum']
        answer = data['note']
        predicted = sess.run(
            predict_op,
            feed_dict={
                X: spectrum,
                Y: utils.one_hot(constants.NUM_NOTES, answer)
            }
        )[0]
        if answer != predicted:
            logging.info("""
                Found incorrect prediction in %d tries:
                    at progress %f in file %s
                    %d instead of %d'
            """,
            step, data['progress'], 'cache/wav_i%d_n%d.wav' % (data['instrument'], answer),
            answer, predicted)
            break
