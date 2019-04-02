
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


def plot_matrix(matrix, ax=None, xlabel=None, ylabel=None, title=None):
    """
    Displays a given matrix as an image.

    Args:
        - matrix: the matrix to be displayed        
        - ax: the matplotlib axis where to overlay the plot. 
          If you create the figure with `fig, fig_ax = plt.subplots()` simply pass `ax=fig_ax`. 
          If you do not explicitily create a figure, then pass no extra argument.  
          In this case the  current axis (i.e. `plt.gca())` will be used        
    """
    if ax is None:
        ax = plt.gca()

    # round down
    matrix = np.round(matrix, decimals=2)

    # image
    handle = ax.imshow(matrix, cmap=plt.get_cmap('summer'), aspect="equal")

    # colorbar
    plt.colorbar(handle)

    # labels
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    # text
    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            plt.text(x, y, matrix[x, y], horizontalalignment='center')


def nearest_centroid_classify(train__set, labels, f1=0, f2=1):

    xs = train__set[:, f1]
    ys = train__set[:, f2]

    test_points = np.array([np.array([x, y]) for (x, y) in zip(xs, ys)])

    classes = []
    mus = get_centroids(xs, ys, labels)

    for i, point in enumerate(test_points):
        ds = []
        for mu in mus:

            d = np.linalg.norm(point-mu)
            ds.append(d)

        min_d = np.amin(ds)

        index = np.where(ds == min_d)

        c = index[0][0] + 1

        classes.append(c)

    return classes


def get_centroids(train_xs, train_ys, labels):
    n_classes = np.unique(labels).size

    points = [[np.array([x, y]), label] for (x, y, label)
              in zip(train_xs, train_ys, labels)]

    a = np.array(points)

    mus = []

    for i in range(n_classes):
        c = i + 1
        arr = np.array(a[a[:, 1] == c])
        mu = np.mean(arr[:, 0])
        mus.append(mu)

    mus = np.array(mus)

    return mus


def calculate_accuracy(gt_labels, pred_labels):
    count = 0
    for (gt, predicted) in zip(gt_labels, pred_labels):
        if (gt == predicted):
            count = count + 1
    return count / gt_labels.size


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


def plot_knn_confusion_matrix(train_set, train_labels, test_set, k):
    return []


def plot_feature_selection_scatter(train_set, train_labels):
    n_features = train_set.shape[1]

    fig, ax = plt.subplots(n_features, n_features, squeeze=False)

    plt.rc('figure', figsize=(12, 8), dpi=110)
    plt.rc('font', size=12)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99,
                        bottom=0.01, wspace=0.2, hspace=0.4)

    class_1_colour = r'#3366ff'
    class_2_colour = r'#cc3300'
    class_3_colour = r'#ffc34d'

    class_colours = [class_1_colour, class_2_colour, class_3_colour]

    for i in range(n_features):
        for j in range(n_features):
            xs = train_set[:, i]
            ys = train_set[:, j]
            colors = [class_colours[label - 1] for label in train_labels]

            ax[i, j].set_yticklabels([])
            ax[i, j].set_xticklabels([])

            # plot
            ax[i, j].scatter(xs, ys, c=colors, s=1)
            # ax[i, j].set_title('Feature %s vs %s' % (i + 1, j + 1))


def compare(train_set, train_labels):

    fig, ax = plt.subplots(2, 2, squeeze=False)

    plt.rc('figure', figsize=(12, 8), dpi=110)
    plt.rc('font', size=12)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99,
                        bottom=0.01, wspace=0.2, hspace=0.4)

    class_1_colour = r'#3366ff'
    class_2_colour = r'#cc3300'
    class_3_colour = r'#ffc34d'

    class_colours = [class_1_colour, class_2_colour, class_3_colour]

    xs = train_set[:, 0]
    ys = train_set[:, 6]
    colors = [class_colours[label - 1] for label in train_labels]

    ax[0, 0].set_yticklabels([])
    ax[0, 0].set_xticklabels([])
    # plot
    ax[0, 0].scatter(xs, ys, c=colors, s=10)
    ax[0, 0].set_title('Feature 0 vs 6')

    xs = train_set[:, 0]
    ys = train_set[:, 11]
    colors = [class_colours[label - 1] for label in train_labels]

    ax[0, 1].set_yticklabels([])
    ax[0, 1].set_xticklabels([])
    # plot
    ax[0, 1].scatter(xs, ys, c=colors, s=10)
    ax[0, 1].set_title('Feature 0 vs 11')

    xs = train_set[:, 6]
    ys = train_set[:, 5]
    colors = [class_colours[label - 1] for label in train_labels]

    ax[1, 0].set_yticklabels([])
    ax[1, 0].set_xticklabels([])
    # plot
    ax[1, 0].scatter(xs, ys, c=colors, s=10)
    ax[1, 0].set_title('Feature 6 vs 5')

    xs = train_set[:, 10]
    ys = train_set[:, 6]
    colors = [class_colours[label - 1] for label in train_labels]

    ax[1, 1].set_yticklabels([])
    ax[1, 1].set_xticklabels([])
    # plot
    ax[1, 1].scatter(xs, ys, c=colors, s=10)
    ax[1, 1].set_title('Feature 10 vs 6')
