#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.contrib.layers import flatten
import pickle
import os
import argparse
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt

def _compute_output_size(input_size, kernal_size, padding_size, stride):
    return (input_size - kernal_size + 2 * padding_size) // stride + 1


def read_data(datadir):
    data = {}
    for filename in ['train', 'valid', 'test']:
        with open(os.path.join(datadir, filename + '.p'), 'rb') as f:
            data[filename ]= pickle.load(f)
    return data


class LeNet(object):

    def __init__(self, X: tf.Variable, y: tf.Variable, num_labels: int, mean: float=0.0, stddev: float=0.1):
        self._X = X
        self._y = y
        self._num_labels = num_labels
        self._one_hot_y = tf.one_hot(self._y, self._num_labels)
        self._mean = mean
        self._stddev = stddev
        self._initialize()

    def _initialize(self, configure=None):
        ''' Initialize cnn definition and opetations
        '''
        if configure is None:
            configure = {}
        
        # Convolution layer 1
        num_filter1 = configure.get('num_filter1', 6)
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, num_filter1), mean=self._mean, stddev=self._stddev))
        conv1_b = tf.Variable(tf.zeros(num_filter1))
        self._conv1 = tf.nn.conv2d(self._X, conv1_W, strides=(1, 1, 1, 1), padding='VALID') + conv1_b
        self._conv1 = tf.nn.relu(self._conv1)
        self._conv1 = tf.nn.max_pool(self._conv1, ksize=[1, 2, 2, 1], strides=(1, 2, 2, 1), padding='VALID', name='conv1')

        # Convolution layer 2
        num_filter2 = configure.get('num_filter2', 16)
        conv2_W = tf.Variable(tf.truncated_normal(shape=[5, 5, num_filter1, num_filter2], mean=self._mean, stddev=self._stddev))
        conv2_b = tf.Variable(tf.zeros(num_filter2))
        self._conv2 = tf.nn.conv2d(self._conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        self._conv2 = tf.nn.relu(self._conv2)
        self._conv2 = tf.nn.max_pool(self._conv2, ksize=[1, 2, 2, 1], strides=(1, 2, 2, 1), padding='VALID', name='conv2')

        self._fc0 = flatten(self._conv2)

        num_fc1 = configure.get('num_fc1', 120)
        self._fc1_W = tf.Variable(tf.truncated_normal(shape=(self._fc0.shape[-1].value, num_fc1), mean=self._mean, stddev=self._stddev), name='fc1')
        self._fc1_b = tf.Variable(tf.zeros(num_fc1))
        self._fc1 = tf.matmul(self._fc0, self._fc1_W) + self._fc1_b

        num_fc2 = configure.get('num_fc1', 84)
        self._fc2_W = tf.Variable(tf.truncated_normal(shape=(num_fc1, num_fc2), mean=self._mean, stddev=self._stddev), name='fc2')
        self._fc2_b = tf.Variable(tf.zeros(num_fc2))
        self._fc2 = tf.matmul(self._fc1, self._fc2_W) + self._fc2_b

        num_fc3 = self._num_labels
        self._fc3_W = tf.Variable(tf.truncated_normal(shape=(num_fc2, num_fc3), mean=self._mean, stddev=self._stddev), name='fc3')
        self._fc3_b = tf.Variable(tf.zeros(num_fc3))
        self._fc3 = tf.matmul(self._fc2, self._fc3_W) + self._fc3_b
    
        # Convert logits int0 proper probablity expression
        self._cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self._one_hot_y, logits=self.logits)

        self._learn_rate = configure.get('learn_rate', 0.001)

        # Optimization definitions
        self._optimizer = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
        self._loss_operation = tf.reduce_mean(self._cross_entropy)
        self._training_operation = self._optimizer.minimize(self._loss_operation)

        # correct prediction for evaluation
        self._correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self._one_hot_y, 1))
        self._accuracy_operation = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))

        self._saver = tf.train.Saver()

    
    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def logits(self):
        return self._fc3
    
    @property
    def accuracy_operation(self):
        return self._accuracy_operation
    
    @property
    def train_operation(self):
        return self._training_operation
    
    @property
    def saver(self):
        return self._saver

    def __call__(self, X_data, y_data, batch_size: int):
        return self.evaluate(X_data, y_data, batch_size)

    def evaluate(self, X_data, y_data, batch_size: int):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, batch_size):
            batch_x, batch_y = X_data[offset: offset + batch_size], y_data[offset: offset + batch_size]
            accuracy = sess.run(self.accuracy_operation, feed_dict={ self.X: batch_x, self.y: batch_y })
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    def output_features(self):
        with tf.Session() as session:
            dummy_input = np.random.random((1, 32, 32, 3))
            dummy_output = [1]
            convfeatures1 = self._conv1.eval(feed_dict={ self.X: dummy_input, self.y: dummy_output })
            convfeatures2 = self._conv2.eval(feed_dict={ self.X: convfeatures1 })
        return convfeatures1, convfeatures2



def run(data, train=False, nepochs=10, learn_rate=0.001, batch_size=128):

    X_train = data['train']['features']
    y_train = data['train']['labels']

    X_validation = data['valid']['features']
    y_validation = data['valid']['labels']

    num_labels = y_train.ptp() + 1 - y_train.min()

    image_shape = X_train[0].shape

    X_var = tf.placeholder(tf.float32, (None, *image_shape))
    y_var = tf.placeholder(tf.int32, (None))

    net = LeNet(X_var, y_var, num_labels)

    if not train:
        with tf.Session() as session:
            net.saver.restore(session, './traffic_sign_classifiter.ckpt')
            graph = session.graph
            conv1 = graph.get_tensor_by_name('conv1:0')
            conv2 = graph.get_tensor_by_name('conv2:0')

            plt.ion()
            plt.figure()
            cnt = 0
            total_cnt = 10
            for i in range(total_cnt):
                index = np.random.randint(0, len(X_train))
                activations1 = conv1.eval(feed_dict={ X_var: X_train[index: index + 1] })
                plt.subplot(total_cnt, activations1.shape[-1] + 1, cnt + 1)
                cnt += 1
                plt.imshow(X_train[index])
                for j in range(activations1.shape[-1]):
                    plt.subplot(total_cnt, activations1.shape[-1] + 1, cnt + 1)
                    plt.imshow(activations1[0][:, :, j])
                    cnt += 1
            plt.show(block=False)


            plt.ion()
            plt.figure()
            cnt = 0
            total_cnt = 10
            for i in range(total_cnt):
                index = np.random.randint(0, len(X_train))
                activations2 = conv2.eval(feed_dict={ X_var: X_train[index: index + 1] })
                plt.subplot(total_cnt, activations2.shape[-1] + 1, cnt + 1)
                cnt += 1
                plt.imshow(X_train[index])
                for j in range(activations2.shape[-1]):
                    plt.subplot(total_cnt, activations2.shape[-1] + 1, cnt + 1)
                    plt.imshow(activations2[0][:, :, j])
                    cnt += 1
            # plt.tight_layout()
            plt.show(block=False)

            embed()
        return

    # Train
    from sklearn.utils import shuffle
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        for i in range(nepochs):

            X_train, y_train = shuffle(X_train, y_train)

            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_X = X_train[offset: end]
                batch_y = y_train[offset: end]
                session.run(net.train_operation, feed_dict={ net.X: batch_X, net.y: batch_y })
            validation_accuracy = net(X_validation, y_validation, batch_size)
            print("EPOCH {} ... Validation Accuracy = {:.3f}".format(i + 1, validation_accuracy))
        
        net.saver.save(session, './traffic_sign_classifiter.ckpt')
        print("Model saved")

    embed()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepochs', type=int, dest='nepochs', default=10)
    parser.add_argument('--learn-rate', type=float, dest='learn_rate', default=0.001)
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)
    parser.add_argument('--train', action='store_true', dest='train')

    options = parser.parse_args()

    data = read_data(os.path.join(os.getcwd(), 'traffic_data'))
    run(data, options.train, options.nepochs, options.learn_rate, options.batch_size)
