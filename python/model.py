#!/usr/bin/env python3
import tensorflow
import tensorflow.compat.v1 as tf
from IPython import embed

class Net(object):
    ''' CNN base
    '''
    _X = None
    _y = None
    _num_labels = None
    _one_hot_y = None
    _mean = None
    _stddev = None
    _saver = None
    _learn_rate = None
    _dropout = None
    _accuracy_operation = None
    _training_operation = None

    def __init__(self, X: tf.Variable, y: tf.Variable, num_labels: int, mean: float=0.0, stddev: float=0.1):
        self._X = X
        self._y = y
        self._num_labels = num_labels
        self._one_hot_y = tf.one_hot(self._y, self._num_labels)
        self._mean = mean
        self._stddev = stddev
        self._learn_rate = 0.001
        self._dropout = tf.placeholder(tf.float32, name='dropout')  # dropout rate
        self.init_structure()

    def init_structure(self):
        pass

    def _get_initialized_var(self, shape):
        return tf.truncated_normal(shape=shape, mean=self._mean, stddev=self._stddev)

    def _create_conv_layer(self, input_var, num_output_features, ksize, strides, name, use_subsample=True):
        num_input_features = input_var.shape[-1].value

        conv_W = tf.Variable(self._get_initialized_var(shape=(ksize, ksize, num_input_features, num_output_features)))
        conv_b = tf.Variable(tf.zeros(num_output_features))

        conv = tf.nn.conv2d(input_var, conv_W, strides=[1, strides, strides, 1], padding='VALID') + conv_b
        conv = tf.nn.relu(conv)
        print('shape=%s before max_pool' % (conv.shape))
        if use_subsample:
            conv = tf.nn.max_pool(conv, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='VALID', name=name)

        return conv, conv_W

    def _create_full_connected_layer(self, input_var, num_output, name):
        num_input = input_var.shape[-1].value
        fc_W = tf.Variable(self._get_initialized_var(shape=(num_input, num_output)), name=name + '_W')
        fc_b = tf.Variable(tf.zeros(num_output), name=name + '_b')
        fc = tf.nn.xw_plus_b(input_var, fc_W, fc_b)
        return fc, fc_W

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def dropout(self):
        return self._dropout

    @property
    def logits(self):
        return self._logits

    @property
    def accuracy_operation(self):
        return self._accuracy_operation

    @property
    def train_operation(self):
        return self._training_operation

    @property
    def saver(self):
        return self._saver

    def __call__(self, session, X_data, y_data, keep_prob: float, batch_size: int):
        return self.evaluate(session, X_data, y_data, keep_prob, batch_size)

    def train(self, session, X_data, y_data, keep_prob):
        session.run(self._training_operation,
                    feed_dict={self.X: X_data, self.y: y_data, self.dropout: keep_prob})

    def evaluate(self, session, X_data, y_data, keep_prob, batch_size: int):
        num_examples = len(X_data)
        total_accuracy = 0
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset: offset + batch_size], y_data[offset: offset + batch_size]
            accuracy = session.run(self.accuracy_operation,
                                   feed_dict={self.X: batch_x, self.y: batch_y, self.dropout: keep_prob})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples
    
    def save(self, session):
        self.saver.save(session, './%s.ckpt' % self)
    
    def load(self, session):
        self.saver.restore(session, './%s.ckpt' % self)


class LeNet5(Net):

    def __str__(self):
        return 'LeNet5'

    def init_structure(self):
        self._conv1, conv1_W = self._create_conv_layer(self._X, num_output_features=6, ksize=5, strides=1, name='conv1')
        self._conv2, conv2_W = self._create_conv_layer(self._conv1, num_output_features=16, ksize=5, strides=1, name='conv2')
        self._fc0 = tf.layers.Flatten()(self._conv2)
        self._fc1, fc1_W = self._create_full_connected_layer(self._fc0, 120, 'fc1')
        self._fc1 = tf.nn.relu(self._fc1)
        self._fc2, fc2_W = self._create_full_connected_layer(self._fc1, 84, 'fc2')
        self._fc2 = tf.nn.relu(self._fc2)
        self._fc3, fc3_W = self._create_full_connected_layer(self._fc2, self._num_labels, 'fc3')

        self._logits = self._fc3

        # Convert logits int0 proper probablity expression
        self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self._one_hot_y, logits=self._logits)
        self._loss_operation = tf.reduce_mean(self._cross_entropy)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
        self._training_operation = self._optimizer.minimize(self._loss_operation)

        # correct prediction for evaluation
        self._correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._one_hot_y, 1))
        self._accuracy_operation = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))
        self._saver = tf.train.Saver()

    def __repr__(self):
        return '%s\n%s\n%s\n%s\n%s\n%s\n%s\n' % (self._X, self._conv1, self._conv2, self._fc0, self._fc1, self._fc2, self._fc3)


class MyNet(Net):

    def __str__(self):
        return 'MyNet'

    def init_structure(self):
        #
        # Net definition
        #
        self._conv1, conv1_W = self._create_conv_layer(self._X, num_output_features=6, ksize=5, strides=1, name='conv1')
        self._conv2, conv2_W = self._create_conv_layer(self._conv1, num_output_features=16, ksize=5, strides=1, name='conv2')
        self._conv3, conv3_W = self._create_conv_layer(self._conv2, num_output_features=400, ksize=5, strides=1, name='conv3', use_subsample=False)

        self._fc0 = tf.concat([tf.layers.Flatten()(self._conv2), tf.layers.Flatten()(self._conv3)], 1)
        self._fc0 = tf.nn.dropout(self._fc0, keep_prob=self._dropout)

        self._fc1, fc1_W = self._create_full_connected_layer(self._fc0, self._num_labels, name='fc1')

        self._logits = self._fc1

        #
        # Training pipeline
        #
        beta = 0.0001
        # Add all weights to regularization term
        reg_term = tf.nn.l2_loss(conv1_W) + \
                   tf.nn.l2_loss(conv2_W) + \
                   tf.nn.l2_loss(conv3_W) + \
                   tf.nn.l2_loss(fc1_W)

        # Convert logits into proper probablity expression
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
        self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self._one_hot_y, logits=self._logits)
        self._loss_operation = tf.reduce_mean(self._cross_entropy + beta * reg_term)
        self._training_operation = self._optimizer.minimize(self._loss_operation)

        #
        # Evaluation pipeline
        #
        # correct prediction for evaluation
        self._correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._one_hot_y, 1))
        self._accuracy_operation = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))
        self._saver = tf.train.Saver()

    def __repr__(self):
        return '%s\n%s\n%s\n%s\n%s\n' % (self._X, self._conv1, self._conv2, self._fc0, self._fc1)
