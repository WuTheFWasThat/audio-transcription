from tensorflow.core.framework import summary_pb2
import numpy as np

def make_summary(name, val):
    return summary_pb2.Summary(value=[summary_pb2.Summary.Value(tag=name,
                                                                simple_value=val)])

def one_hot(n, i):
    assert i < n
    assert i >= 0
    return np.reshape([int(x == i) for x in xrange(n)], (1, -1))

