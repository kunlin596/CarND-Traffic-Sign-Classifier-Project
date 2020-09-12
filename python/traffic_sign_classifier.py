import tensorflow.compat.v1 as tf
from tensorflow.contrib.layers import flatten
import pickle
import os
import argparse

def _compute_output_size(input_size, kernal_size, padding_size, stride):
    return (input_size - kernal_size + 2 * padding_size) // stride + 1


def read_data(datadir):
    data = {}
    for filename in ['train', 'valid', 'test']:
        with open(os.path.join(datadir, filename + '.p'), 'rb') as f:
            data[filename ]= pickle.load(f)
    return data


class LeNet(object):

    def get_logits(self, X: tf.Variable, num_labels: int, mean: float=0.0, stddev: float=0.1):
        # Convolution layer 1
        num_filter1 = 6
        conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, num_filter1), mean=mean, stddev=stddev))
        conv1_b = tf.Variable(tf.zeros(num_filter1))
        self._conv1 = tf.nn.conv2d(X, conv1_W, strides=(1, 1, 1, 1), padding='VALID') + conv1_b
        self._conv1 = tf.nn.relu(self._conv1)
        self._conv1 = tf.nn.max_pool(self._conv1, ksize=[1, 2, 2, 1], strides=(1, 2, 2, 1), padding='VALID')

        # Convolution layer 2
        num_filter2 = 16
        conv2_W = tf.Variable(tf.truncated_normal(shape=[5, 5, num_filter1, num_filter2], mean=mean, stddev=stddev))
        conv2_b = tf.Variable(tf.zeros(num_filter2))
        self._conv2 = tf.nn.conv2d(self._conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
        self._conv2 = tf.nn.relu(self._conv2)
        self._conv2 = tf.nn.max_pool(self._conv2, ksize=[1, 2, 2, 1], strides=(1, 2, 2, 1), padding='VALID')

        self._fc0 = flatten(self._conv2)

        self._fc1_W = tf.Variable(tf.truncated_normal(shape=(self._fc0.shape[-1].value, 200), mean=mean, stddev=stddev))
        self._fc1_b = tf.Variable(tf.zeros(200))
        self._fc1 = tf.matmul(self._fc0, self._fc1_W) + self._fc1_b

        self._fc2_W = tf.Variable(tf.truncated_normal(shape=(200, 100), mean=mean, stddev=stddev))
        self._fc2_b = tf.Variable(tf.zeros(100))
        self._fc2 = tf.matmul(self._fc1, self._fc2_W) + self._fc2_b

        self._fc3_W = tf.Variable(tf.truncated_normal(shape=(100, num_labels), mean=mean, stddev=stddev))
        self._fc3_b = tf.Variable(tf.zeros(num_labels))
        self._fc3 = tf.matmul(self._fc2, self._fc3_W) + self._fc3_b

        return self._fc3


def evaluate(X_data, y_data, x, y, batch_size, accuracy_operation):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset: offset + batch_size], y_data[offset: offset + batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={ x: batch_x, y: batch_y })
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def run(data, nepochs=10, learn_rate=0.001, batch_size=128):
    X_train = data['train']['features']
    y_train = data['train']['labels']

    X_validation = data['valid']['features']
    y_validation = data['valid']['labels']

    num_labels = y_train.ptp()
    min_label = y_train.min()

    x = tf.placeholder(tf.float32, (None, 32, 32, 3))
    y = tf.placeholder(tf.int32, (None))
    one_hot_y = tf.one_hot(y, num_labels)

    net = LeNet()
    logits = net.get_logits(x, num_labels)

    # Convert logirs intto proper probablity expression
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)

    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)

    #  Different operations
    loss_operation = tf.reduce_mean(cross_entropy)

    training_operation = optimizer.minimize(loss_operation)

    # correct prediction for evaluation
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))

    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

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
                session.run(training_operation, feed_dict={x: batch_X, y: batch_y})
            
            validation_accuracy = evaluate(X_validation, y_validation, x, y, batch_size, accuracy_operation)
            print("EPOCH {} ... Validation Accuracy = {:.3f}".format(i + 1, validation_accuracy))
        
        saver.save(session, './traffic_sign_classifiter')
        print("Model saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepochs', type=int, dest='nepochs', default=10)
    parser.add_argument('--learn-rate', type=float, dest='learn_rate', default=0.001)
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)

    options = parser.parse_args()

    data = read_data(os.path.join(os.getcwd(), 'traffic_data'))
    run(data, options.nepochs, options.learn_rate, options.batch_size)
