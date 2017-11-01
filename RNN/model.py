import tensorflow as tf
from tensorflow.contrib import rnn
import functools

def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class RNN_Model:
    def __init__(self,inputs,labels,n_inputs=28,n_steps=28,n_hidden_units=128,n_classes = 10):
       self.input=inputs
       self.labels=labels
       self.n_inputs=n_inputs
       self.n_steps=n_steps
       self.n_hidden_units=n_hidden_units
       self.n_classes=n_classes

       self.predict
       self.optimizer
       self.accuracy

    @define_scope
    def predict(self):
        X=self.input
        with tf.name_scope("inlayer"):
            weights_in = tf.Variable(tf.random_uniform([self.n_inputs, self.n_hidden_units], -1.0, 1.0), name="in_w")
            b_in = tf.Variable(tf.constant(0.1, shape=[self.n_hidden_units]), name="in_bias")
            X = tf.reshape(X, [-1, self.n_inputs])
            X_in = tf.matmul(X, weights_in) + b_in
            X_in = tf.reshape(X_in, [-1, self.n_steps, self.n_hidden_units])
        # RNN cell
        with tf.name_scope("RNN_CELL"):
            lstm_cell = rnn.BasicLSTMCell(self.n_hidden_units)
            # _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
            # ouputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state)
            outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, dtype=tf.float32)
            # print (states[1].shape)
        # out layer
        with tf.name_scope('outlayer'):
            weights_out = tf.Variable(tf.random_uniform([self.n_hidden_units, self.n_classes], -1.0, 1.0), name="out_w")
            b_out = tf.Variable(tf.constant(0.1, shape=[self.n_classes]), name="out_bias")
            logist = tf.matmul(states[1], weights_out) + b_out
        return logist

    @define_scope
    def optimizer(self,lr=0.0001):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict, labels=self.labels))
        train_op = tf.train.AdamOptimizer(lr).minimize(cost)
        return train_op

    @define_scope
    def accuracy(self):
        correct_pred = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.predict, labels=self.labels))
        return accuracy,cost
