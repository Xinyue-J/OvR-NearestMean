
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def plot(training, label_train, mean1, mean2, mean3):
    # Plot the decision boundaries and data points for minimum distance to
    # OvR class mean classifier-hw2
    # training: training data
    # label_train: class labels correspond to training data
    # sample_mean: mean vector for each class
    # Total number of classes

    # Set the feature range for plotting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc),
                         np.arange(yrange[0], yrange[1] + inc / 100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'),
                    y.reshape(y.shape[0] * y.shape[1], 1, order='F')))  # make (x,y) pairs as a bunch of row vectors.

    # distance measure evaluations for each (x,y) pair.
    dist1 = cdist(xy, mean1)
    pred_label1 = np.argmin(dist1, axis=1)  # class1:0

    dist2 = cdist(xy, mean2)
    pred_label2 = np.argmin(dist2, axis=1)  # class2:0

    dist3 = cdist(xy, mean3)
    pred_label3 = np.argmin(dist3, axis=1)  # class3:0

    pred_label=np.zeros(len(pred_label1))
    for i in range(0, len(pred_label1)):
        if pred_label1[i] == 0 and pred_label2[i] == 1 and pred_label3[i] == 1:  # class 1
            pred_label[i] = 0

        elif pred_label1[i] == 1 and pred_label2[i] == 0 and pred_label3[i] == 1:  # class 2
            pred_label[i] = 1
        elif pred_label1[i] == 1 and pred_label2[i] == 1 and pred_label3[i] == 0:  # class 3
            pred_label[i] = 2
        else:
            pred_label[i] = 3


    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    # show the image, give each coordinate a color according to its class label
    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower')

    # plot the class training data.
    plt.plot(training[label_train == 1, 0], training[label_train == 1, 1], 'rx')
    plt.plot(training[label_train == 2, 0], training[label_train == 2, 1], 'go')
    plt.plot(training[label_train == 3, 0], training[label_train == 3, 1], 'b*')

    l = plt.legend(('Class 1', 'Class 2', 'Class 3'), loc=2)
    plt.gca().add_artist(l)

    # plot the class mean vector.
    m1, = plt.plot(mean1[0, 0], mean1[0, 1], 'rd', markersize=12, markerfacecolor='r', markeredgecolor='w')
    m2, = plt.plot(mean2[0, 0], mean2[0, 1], 'gd', markersize=12, markerfacecolor='g', markeredgecolor='w')
    m3, = plt.plot(mean3[0, 0], mean3[0, 1], 'bd', markersize=12, markerfacecolor='b', markeredgecolor='w')

    plt.legend([m1, m2, m3], ['Class 1 Mean', 'Class 2 Mean', 'Class 3 Mean'], loc=4)
    plt.gca().add_artist(l)

    plt.show()
