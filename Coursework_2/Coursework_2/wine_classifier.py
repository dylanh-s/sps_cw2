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
<<<<<<< HEAD
import matplotlib as mpl
import mpl_toolkits.mplot3d as mpl_t
#from mpl_toolkits import Axes3
=======
from sklearn.decomposition import PCA


>>>>>>> PCA

from utilities import load_data, print_features, print_predictions
from feature_select import compare, plot_feature_selection_scatter, plot_matrix, calculate_accuracy

# you may use these colours to produce the scatter plots
CLASS_1_C = r'#3366ff'
CLASS_2_C = r'#cc3300'
CLASS_3_C = r'#ffc34d'

colors = [CLASS_1_C, CLASS_2_C, CLASS_3_C]

MODES = ['feature_sel', 'knn', 'knn_accuracy',
         'knn_confusion' 'alt', 'knn_3d', 'knn_pca', 'knn_pca_evaluation']

# FEATURE SELECT
def feature_selection(train_set, train_labels, **kwargs):
    #compare(train_set, [int(i) for i in train_labels])
    #plot_feature_selection_scatter(train_set, train_labels)
    plt.show()
    # TODO call plt.show() here to plot the confusion matrix for use in report

    # return np.where(matrix == np.amax(matrix))[0]
    return [0, 6]

<<<<<<< HEAD
def feature_sel_3d(train_set, train_labels, **kwargs):
    n_features = train_set.shape[1]

    class_1_colour = r'#3366ff'
    class_2_colour = r'#cc3300'
    class_3_colour = r'#ffc34d'

    class_colours = [class_1_colour, class_2_colour, class_3_colour]
    colors = [class_colours[int(label) - 1] for label in train_labels]

    for i in range(0,n_features):
        fig = plt.figure()
        ax = mpl_t.Axes3D(fig)
        ax.scatter(train_set[:,0], train_set[:,6], train_set[:,i], c=colors)
        ax.set_zlabel("{}".format(i))
        plt.xlabel("0")
        plt.ylabel("6")
        plt.title("{}".format(i))
        plt.show()

=======
# PLOT
# ACCURACY
>>>>>>> PCA
def plot_alt_accuracy( train_set, train_labels, test_set, test_labels):
    pred_labels = alternative_classifier(train_set,train_labels,test_set, 0, 6)
    print(calculate_accuracy(test_labels, pred_labels))


def plot_knn_accuracy(train_set, train_labels, test_set, test_labels):

    ks = [1, 2, 3, 4, 5, 7]
    accs = []

    for k in ks:

        pred_labels = knn(train_set, train_labels, test_set, k, 0, 6)

        a = calculate_accuracy(test_labels, pred_labels)

        accs.append([k, a])

    print(accs)


def plot_feature_selection_accuracy_matrix(train_set, train_labels, test_set, test_labels, k):
    n_features = train_set.shape[1]
    print(train_labels)
    matrix = np.zeros((n_features, n_features))

    # write your code here
    for i in range(n_features):
        for j in range(n_features):

            pred_labels = knn(train_set, train_labels, test_set, k, i, j)

            accuracy = calculate_accuracy(test_labels, pred_labels)

            matrix[i, j] = accuracy

    plot_matrix(matrix, xlabel='feature', ylabel='feature',
                title='Accuracy with different feature combinations')

    plt.show()

    return matrix


def foo(i, j, gt_labels, pred_labels):
    count = 0
    for x in range(pred_labels.size):
        if gt_labels[x] == i:
            if pred_labels[x] == j:
                count += 1
    return count

# CONFUSION MATRIX
def plot_confusion_matrix(gt_labels, pred_labels, ax=None, title=None):
    classes = np.unique(gt_labels)

    counts = np.zeros(classes.size)
    cm = np.zeros((classes.size, classes.size))
    
    for i in range(1, classes.size+1):
        counts[i-1] = np.count_nonzero(gt_labels == i)
    
    for i in range(counts.size):
        for j in range(counts.size):
            cm[i, j] = (foo(i + 1, j + 1, gt_labels, pred_labels))/counts[i]

   
    plot_matrix(cm.transpose(), ax=ax, title=title)

def compare_confusion_matricies(pred1, title1, pred2, title2, test_labels):
    fig, ax = plt.subplots(1, 2)
    
    plot_confusion_matrix(test_labels, pred1, ax=ax[0], title=title1)
    plot_confusion_matrix(test_labels, pred2, ax=ax[1], title=title2)
    




# THIS DOESNT WORK :(
def plot_confusion_matrices(train_set, train_labels, test_set, test_labels, f1, f2):
    ks = np.array([1, 3, 5, 7])

    fig, fig_ax = plt.subplots(ks.size, squeeze=True)

    plt.rc('figure', figsize=(18, 30), dpi=110)
    plt.rc('font', size=12)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9,
                        bottom=0.1, wspace=0.2, hspace=0.4)
    count = 0
    for a in fig_ax:
        pred_labels = np.array(knn(train_set, train_labels, test_set, ks[count], 0, 6))
        cm = plot_confusion_matrix(test_labels, pred_labels)
        plot_matrix(cm, ax=a, title = "k = {}".format(ks[count]))
        count = count + 1
    plt.show()

# SCATTER
def plot_scatter(xs, ys, train_labels, title="Title"):
    c = [colors[int(label) - 1] for label in train_labels]
    plt.scatter(xs, ys, c=c)
    plt.show()

def plot_scatter_comparison(scipy_set, my_set, train_labels):
    cs = [colors[int(label) - 1] for label in train_labels]

    fig, ax = plt.subplots(1, 2)
   
    xs_s = scipy_set[:, 0]
    ys_s = scipy_set[:, 1]
    
    xs_m = my_set[:, 0]
    ys_m = my_set[:, 1]
    
    ax[1].scatter(xs_s, ys_s, c=cs)
    ax[1].set_title('Scipy PCA')

    ax[0].scatter(xs_m, ys_m, c=cs)
    ax[0].set_title('Manual feature select')

    plt.show()


# CLASSIFIERS
# KNN
def knn(train_set, train_labels, test_set, k, f1, f2, **kwargs):
    cols = train_set.shape[1]
    
    # if the train and test sets have already been reduced
    if (cols == 2):
        f1 = 0
        f2 = 1
    
    train_xs = train_set[:, f1]
    train_ys = train_set[:, f2]
    train_points = np.array([np.array([x, y])
                             for (x, y) in zip(train_xs, train_ys)])

    test_xs = test_set[:, f1]
    test_ys = test_set[:, f2]
    test_points = np.array([np.array([x, y])
                            for (x, y) in zip(test_xs, test_ys)])

    classes = []
    for test_p in test_points:
        ds = []
        for i, train_p in enumerate(train_points):
            d = np.linalg.norm(test_p - train_p)
            ds.append([d, train_labels[i]])
        ds = np.array(ds)
        # sort ds by distances
        args = np.argsort(ds, axis=0)

        sort = np.argsort(ds[:, 0])

        ds = np.array([ds[i] for i in sort])

        # get first k elements, cast them to int
        ks = [int(k) for k in ds[:, 1][:k]]

        # get the majority element
        majority_c = Counter(ks).most_common()[0][0]
        classes.append(majority_c)

    return np.array(classes)

def alternative_post_likelihood(mean, var, p, n_classes=3):
    #probability for class c P(x|c) = np.prod ( p(x_i|c) for all features )
    #p(x_i|c) = (1/np.sqrt(2*3.14*vars[i]) * np.exp(-0.5 * np.power((x - means[i]),2)/vars[i]
    n_features = mean[0].size
    probs = []
    #for each class
    for c in range(0,n_classes):
        prod = 1
        #for each feature
        for j in range(0,n_features):
            prod = prod * (1/np.sqrt(2*3.14*var[c][j]) * np.exp(-0.5 * np.power((p[j] - mean[c][j]),2)/var[c][j]))
        probs.append(prod)
    return probs

def alternative_classifier(train_set, train_labels, test_set, f1, f2, **kwargs):
    train_xs = train_set[:, f1]
    train_ys = train_set[:, f2]
    train_points = np.array([np.array([x, y])
                             for (x, y) in zip(train_xs, train_ys)])

    test_xs = test_set[:, f1]
    test_ys = test_set[:, f2]
    test_points = np.array([np.array([x, y])
                            for (x, y) in zip(test_xs, test_ys)])

    classes = np.unique(train_labels)

    #priors of each class
    counts = np.zeros(classes.size)
    priors = np.zeros(classes.size)
    for i in range(1, classes.size+1):
        counts[i-1] = np.count_nonzero(train_labels == i)
        priors[i-1] = counts[i-1]/train_labels.size
    #print(priors)

    #mean of each class
    x_means = [np.mean(train_xs[train_labels == c]) for c in classes]
    y_means = [np.mean(train_ys[train_labels == c]) for c in classes]
    means = np.array([np.array([x, y]) for (x, y) in zip(x_means, y_means)])
    #print(means)

    #var of each class
    x_vars = [np.var(train_xs[train_labels == c]) for c in classes]
    y_vars = [np.var(train_ys[train_labels == c]) for c in classes]
    vars = np.array([np.array([x, y]) for (x, y) in zip(x_vars, y_vars)])

    predictions = []
    #classify points
    for test_p in test_points:
        options = np.ones(classes.size)
        post_prob_c = alternative_post_likelihood(means,vars,test_p)
        for i in range(0,classes.size):
            options[i] = priors[i] * post_prob_c[i]
        predictions.append(np.argmax(options) + 1)
    return predictions


def knn_three_features(train_set, train_labels, test_set, k, f1, f2, f3, **kwargs):
    train_xs = train_set[:, f1]
    train_ys = train_set[:, f2]
    train_zs = train_set[:, f3]
    train_points = np.array([np.array([x, y, z])
                             for (x, y, z) in zip(train_xs, train_ys, train_zs)])

    test_xs = test_set[:, f1]
    test_ys = test_set[:, f2]
    test_zs = test_set[:, f3]
    test_points = np.array([np.array([x, y, z])
                            for (x, y, z) in zip(test_xs, test_ys, test_zs)])

    classes = []
    for test_p in test_points:
        ds = []
        for i, train_p in enumerate(train_points):
            d = np.linalg.norm(test_p - train_p)
            ds.append([d, train_labels[i]])
        ds = np.array(ds)
        # sort ds by distances
        args = np.argsort(ds, axis=0)

        sort = np.argsort(ds[:, 0])

        ds = np.array([ds[i] for i in sort])

        # get first k elements, cast them to int
        ks = [int(k) for k in ds[:, 1][:k]]

        # get the majority element
        majority_c = Counter(ks).most_common()[0][0]
        classes.append(majority_c)

    return classes



def knn_pca(train_set, train_labels, test_set, k, n_components=2, **kwargs):
    # write your code here and make sure you return the predictions at the end of
    # the function

    # get the manuall selected set
    feature_sel_set = np.array(list(zip(train_set[:, 0], train_set[:, 6])))

    pca = PCA(n_components=n_components)
    model = pca.fit(train_set)
    scipy_set = pca.transform(train_set)
    test_set = pca.transform(test_set)

    # plotting the pca vs manually selected
    # plot_scatter_comparison(scipy_set, feature_sel_set, train_labels)
 
    return knn(scipy_set, train_labels, test_set, k, 0, 6 )

def manual_vs_pca(train_set, train_labels, test_set, test_labels):
    ks = [1, 2, 3, 4, 5, 7]
    accs = []

    for k in ks:
        man_pred = knn(train_set, train_labels, test_set, k , 0 , 6).astype(np.int)
        pca_pred = knn_pca(train_set, train_labels, test_set, k, n_components=2).astype(np.int)

        man_acc = calculate_accuracy(test_labels, man_pred)
        pca_acc = calculate_accuracy(test_labels, pca_pred)

        accs.append([k, man_acc, pca_acc, man_pred, pca_pred])
    
    for acc in accs:
        # print accuracies
        print("Accuracies for %s cluster(s):\n \t - %s for manual selection \n\t - %s for pca" % (acc[0], acc[1], acc[2]))

        # display confusion matricies alongisde each other
        compare_confusion_matricies(acc[3], "Manual", acc[4], "PCA", test_labels)
    



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
        plot_feature_selection_accuracy_matrix(train_set, train_labels, test_set, test_labels, 3)
        print_features(selected_features)
    elif mode == 'knn':
        predictions = knn(train_set, train_labels, test_set, args.k, 0, 6)
        print_predictions(predictions)
    elif mode == 'knn_accuracy':
        plot_knn_accuracy(
            train_set, train_labels, test_set, test_labels)
    elif mode == 'knn_confusion':
        pred_labels = np.array(
            knn(train_set, train_labels, test_set, 1, 0, 6))
        plot_confusion_matrix(test_labels, pred_labels)
        pred_labels = np.array(
            knn(train_set, train_labels, test_set, 3, 0, 6))
        plot_confusion_matrix(test_labels, pred_labels)
        pred_labels = np.array(
            knn(train_set, train_labels, test_set, 5, 0, 6))
        plot_confusion_matrix(test_labels, pred_labels)
    elif mode == 'alt':
        predictions = alternative_classifier(train_set, train_labels, test_set, 0, 6)
        print_predictions(predictions)
    elif mode == 'alt_accuracy':
        predictions = np.array(alternative_classifier(train_set, train_labels, test_set, 0, 6))
        plot_alt_accuracy(train_set, train_labels, test_set, test_labels)
        plot_confusion_matrix(test_labels, predictions)
    elif mode == 'feature_sel_3d':
        feature_sel_3d(train_set, train_labels)
    elif mode == 'knn_3d':
        features = feature_sel_3d(train_set, train_labels)
        predictions = knn_three_features(
            train_set, train_labels, test_set, args.k, 0, 6, 1)
        print(calculate_accuracy(test_labels, predictions))
        print_predictions(predictions)
    elif mode == 'knn_pca':
        prediction = knn_pca(train_set, train_labels, test_set, args.k)
        print_predictions(prediction)
    elif mode == 'knn_pca_evaluation':
        manual_vs_pca(train_set, train_labels, test_set, test_labels)
        plt.show()
    else:
        raise Exception(
            'Unrecognised mode: {}. Possible modes are: {}'.format(mode, MODES))
