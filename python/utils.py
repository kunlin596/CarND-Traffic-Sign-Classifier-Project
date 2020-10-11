#!/usr/bin/env python3

import matplotlib.pyplot as plt
import copy
import numpy as np
from IPython import embed
import cv2

#
# Data preprocessing
#

def rgb2gray(X):
    X = copy.deepcopy(X)
    X = np.sum(X / 3, axis=3, keepdims=True).round().astype(np.uint8)
    return X

def normalization(X):
    return X / 255 * 2.0 - 1.0

def rotate(image):
    angle_range = 10.0
    angle = np.random.random() * angle_range - angle_range / 2
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def scale(image):
    rows,cols = image.shape[:2]
    px = np.random.randint(-5, 5)
    pts1 = np.float32([[px, px],[rows - px, px], [px, cols - px],[rows - px, cols - px]])
    pts2 = np.float32([[0, 0],[rows, 0],[0, cols],[rows, cols]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    new_image = cv2.warpPerspective(image, M, (rows, cols))
    return new_image

def color(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

augmentation_funcs = [
    rotate,
    scale,
    color
]

def augment_data(X, y, target_number):
    '''Use basic image manipulation to augment the data set'''
    original_length = len(X)
    while len(X) < target_number:
        # Randomly select a image, transform it and push it back to array
        source_index = np.random.randint(0, original_length)
        source_X = X[source_index]
        source_y = y[source_index]

        augmentation_func = np.random.choice(augmentation_funcs)
        new_X = augmentation_func(source_X)
        X = np.append(X, new_X[np.newaxis, ...], axis=0)
        y = np.append(y, source_y[np.newaxis, ...], axis=0)
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
