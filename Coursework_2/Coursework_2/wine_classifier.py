#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Skeleton code for CW2 submission. 
We suggest you implement your code in the provided functions
Make sure to use our print_features() and print_predictions() functions
to print your results
"""

from __future__ import print_function
from collections import Counter

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from utilities import load_data, print_features, print_predictions
from feature_select import compare, plot_feature_selection_scatter, plot_feature_selection_confusion

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

MODES = ['feature_sel', 'knn', 'alt', 'knn_3d', 'knn_pca']


def feature_selection(train_set, train_labels, **kwargs):

    compare(train_set, [int(i) for i in train_labels])

    plt.show()
    # TODO call plt.show() here to plot the confusion matrix for use in report

    # return np.where(matrix == np.amax(matrix))[0]
    return [0, 6]


def knn(train_set, train_labels, test_set, k, **kwargs):

    train_xs = train_set[:, 0]
    train_ys = train_set[:, 6]
    train_points = np.array([np.array([x, y])
                             for (x, y) in zip(train_xs, train_ys)])

    test_xs = train_set[:, 0]
    test_ys = train_set[:, 6]
    test_points = np.array([np.array([x, y])
                            for (x, y) in zip(train_xs, train_ys)])

    classes = []
    for test_p in test_points:
        ds = []
        for i, train_p in enumerate(train_points):
            d = np.linalg.norm(test_p - train_p)
            ds.append([d, train_labels[i]])

        ds = np.array(ds)
        ds = np.sort(ds, axis=0)
        kds = ds[:k][:, 1]
        print(kds)
        classes.append(c)

    return ds[:k][:, 1]
    # mus = get_centroids(xs, ys, train_labels)

    # for i, point in enumerate(test_points):
    #     ds = []
    #     for mu in mus:
    #         d = np.linalg.norm(point-mu)
    #         ds.append(d)

    #     min_d = np.amin(ds)

    #     index = np.where(ds == min_d)

    #     c = index[0][0] + 1

    #     classes.append(c)

    return classes


def alternative_classifier(train_set, train_labels, test_set, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_three_features(train_set, train_labels, test_set, k, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function
    return []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', nargs=1, type=str,
                        help='Running mode. Must be one of the following modes: {}'.format(MODES))
    parser.add_argument('--k', nargs='?', type=int, default=1,
                        help='Number of neighbours for knn')
    parser.add_argument('--train_set_path', nargs='?', type=str,
                        default='data/wine_train.csv', help='Path to the training set csv')
    parser.add_argument('--train_labels_path', nargs='?', type=str,
                        default='data/wine_train_labels.csv', help='Path to training labels')
    parser.add_argument('--test_set_path', nargs='?', type=str,
                        default='data/wine_test.csv', help='Path to the test set csv')
    parser.add_argument('--test_labels_path', nargs='?', type=str,
                        default='data/wine_test_labels.csv', help='Path to the test labels csv')

    args = parser.parse_args()
    mode = args.mode[0]

    return args, mode


if __name__ == '__main__':
    args, mode = parse_args()  # get argument from the command line

    # load the data
    train_set, train_labels, test_set, test_labels = load_data(train_set_path=args.train_set_path,
                                                               train_labels_path=args.train_labels_path,
                                                               test_set_path=args.test_set_path,
                                                               test_labels_path=args.test_labels_path)
    if mode == 'feature_sel':
        selected_features = feature_selection(train_set, train_labels)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set)
        print_predictions(predictions)
    elif mode == 'knn_3d':
        predictions = knn_three_features(
            train_set, train_labels, test_set, args.k)
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    else:
        raise Exception(
            'Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
