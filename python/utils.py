#!/usr/bin/env python3

import matplotlib.pyplot as plt
import copy
import numpy

#
# Data preprocessing
#

def rgb2gray(X):
    X = copy.deepcopy(X)
    X = numpy.sum(X / 3, axis=3, keepdims=True).round().astype(numpy.uint8)
    return X

def normalization(X):
    return (X - 128) / 128

def augment_data(X, y, target_number):
    ''' Use basic image manipulation to augment the data set'''
    return X, y

#
# Data visualization
#

def plot_label_distribution(labels):
    pass

def random_image_visualization(X, y):
    import random
    plt.figure(figsize=(20, 10))
    num_rows = 5
    num_cols = 10
    for row in range(num_rows):
        for col in range(num_cols):
            plt.subplot(num_rows, num_cols, col + 1 + row * num_cols)
            index = random.randint(0, len(X))
            plt.title('label: %d' % y[index], fontsize=20)
            cmap = None
            if X[index].shape[-1] != 3:
                cmap = 'gray'
            plt.imshow(X[index].squeeze(), cmap=cmap)
    plt.tight_layout()