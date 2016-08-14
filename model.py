from __future__ import absolute_import, division, print_function

import generate

import tensorflow as tf
import numpy as np

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w):
    # NOTE: there is a baked in cost function which performs softmax and cross entropy
    return tf.matmul(X, w)

num_frequencies = 2205
num_notes = 120

X = tf.placeholder("float", [None, num_frequencies]) # create symbolic variables
Y = tf.placeholder("float", [None, num_notes])

w = init_weights([num_frequencies, num_notes])

py_x = model(X, w)

# compute mean cross entropy (softmax is applied internally)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
# construct optimizer
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
# at predict time, evaluate the argmax of the logistic regression
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    # restore from checkpoint
    if False:
        saver.restore(sess, "model.ckpt.something")

    for i in range(10000):
        if i % 100 == 0:
            print('Iteration %d' % i)

            # Save the variables to disk.
            save_path = saver.save(sess, "model.ckpt.%d" % i)
            print("Model saved in file: %s" % save_path)

            for i in range(10):
                (frequencies, answer) = generate.sampleLabeledData()
                predicted = sess.run(predict_op, feed_dict={X: frequencies, Y: answer})[0]
                if np.argmax(answer) == predicted:
                    print('CORRECT!', np.argmax(answer))
                else:
                    print('INCORRECT!', np.argmax(answer), predicted)

        (frequencies, answer) = generate.sampleLabeledData()
        # train
        sess.run(train_op, feed_dict={X: frequencies, Y: answer})

    for i in range(10):
        (frequencies, answer) = generate.sampleLabeledData()
        predicted = sess.run(predict_op, feed_dict={X: frequencies, Y: answer})[0]
        if np.argmax(answer) == predicted:
            print('CORRECT!', np.argmax(answer))
        else:
            print('INCORRECT!', np.argmax(answer), predicted)

    # print(i, np.mean(np.argmax(teY, axis=1) ==
    #                  sess.run(predict_op, feed_dict={X: teX, Y: teY})))
