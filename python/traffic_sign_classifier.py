#!/usr/bin/env python3
import tensorflow
import tensorflow.compat.v1 as tf
import pickle
import os
import sys
import argparse
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet5, MyNet
import utils


def _compute_output_size(input_size, kernal_size, padding_size, stride):
    return (input_size - kernal_size + 2 * padding_size) // stride + 1


def read_data(datadir):
    data = {}
    for filename in ['train', 'valid', 'test']:
        with open(os.path.join(datadir, filename + '.p'), 'rb') as f:
            data[filename] = pickle.load(f)
    
    X_train = data['train']['features']
    y_train = data['train']['labels']

    from collections import Counter 
    count = Counter(y_train)

    target_num_image = count.most_common(1)[0][1] * 0.5
    new_X_train = np.empty(shape=(0, *X_train.shape[1:]), dtype=X_train.dtype)
    new_y_train = np.empty(shape=(0, ), dtype=y_train.dtype)
    for k, v in count.items():
        X_train_ = X_train[y_train == k]
        y_train_ = y_train[y_train == k]
        if v < target_num_image:
            print('Augmenting label=%d, count=%d, target_num_image=%d ...' % (k, v, target_num_image))
            X_train_, y_train_ = utils.augment_data(X_train_, y_train_, target_num_image)
        new_X_train = np.append(new_X_train, X_train_, axis=0)
        new_y_train = np.append(new_y_train, y_train_, axis=0)
    
    data['train']['features'] = np.array(new_X_train)
    data['train']['labels'] = np.array(new_y_train)

    return data


def batch_normalization(X):
    from utils import rgb2gray, normalization
    X = rgb2gray(X)
    X = normalization(X)
    return X


def test(data, net):
    X_test = batch_normalization(data['test']['features'])
    y_test = data['test']['labels']
    accuracies = []
    with tf.Session() as session:
        net.load(session)
        import prettytable
        result_table = prettytable.PrettyTable(['Label', '#data', 'Accuracy'])
        for i in range(y_test.ptp() + 1):
            bool_y_label = (y_test == i)
            X_valid = X_test[bool_y_label]
            y_valid = y_test[bool_y_label]
            accuracy = net(session, X_valid, y_valid, 1, len(y_valid)) 
            result_table.add_row([i, len(y_valid), np.round(accuracy, 3)])
            accuracies.append(accuracy)
        print(result_table)

    print('Mean accuracies' % np.mean(accuracies))
    return accuracies


def _analyze_training_data(X, y):
    num_labels = y.ptp() + 1 - y.min()
    all_cnt = np.zeros(num_labels, dtype=np.uint32)
    for label in np.arange(y.min(), y.max() + 1):
        all_cnt[label] = np.count_nonzero(y == label)


def _run(data, net_cls, nepochs=10, learn_rate=0.001, batch_size=128):

    print('Normalizing training data')
    X_train = batch_normalization(data['train']['features'])
    y_train = data['train']['labels']

    print('Normalizing validation data')
    X_validation = batch_normalization(data['valid']['features'])
    y_validation = data['valid']['labels']

    X_test = batch_normalization(data['test']['features'])
    y_test = data['test']['labels']

    num_labels = y_train.ptp() + 1 - y_train.min()

    image_shape = X_train[0].shape

    X_var = tf.placeholder(tf.float32, shape=(None, *image_shape), name='input_features')
    y_var = tf.placeholder(tf.int32, shape=(None), name='input_labels')

    all_cnt = np.zeros(num_labels, dtype=np.uint32)
    for label in np.arange(y_train.min(), y_train.max() + 1):
        all_cnt[label] = np.count_nonzero(y_train == label)

    net = net_cls(X_var, y_var, num_labels)

    embed()

    # Train
    curr_accuracy = None
    from sklearn.utils import shuffle
    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        num_examples = len(X_train)
        for i in range(nepochs):

            X_train, y_train = shuffle(X_train, y_train)

            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_X, batch_y = X_train[offset: end], y_train[offset: end]
                net.train(session, batch_X, batch_y, 0.75)

            validation_accuracy = net(session, X_validation, y_validation, 1.0, batch_size)
            if curr_accuracy is None:
                curr_accuracy = validation_accuracy
            elif curr_accuracy - validation_accuracy > 0.05:
                print('accuracy droping more than 0.05, breaking training early at #epoch=%d' % i)
                break
            else:
                curr_accuracy = validation_accuracy
            print("EPOCH #{:<2} ... Validation Accuracy = {:.3f}".format(i + 1, validation_accuracy))
        net.save(session)
        print("Model saved")
    embed()

if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.disable_v2_behavior()
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepochs', type=int, dest='nepochs', default=30)
    parser.add_argument('--learn-rate', type=float, dest='learn_rate', default=0.001)
    parser.add_argument('--batch-size', type=int, dest='batch_size', default=128)
    parser.add_argument('--model', type=str, dest='model', default='MyNet')

    options = parser.parse_args()
    embed()

    data = read_data(os.path.join(os.getcwd(), 'traffic_data'))
    print('fnished read and preprocessing data')
    embed()
    _run(data, getattr(sys.modules['model'], options.model), options.nepochs, options.learn_rate, options.batch_size)
