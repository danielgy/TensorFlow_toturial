
import tensorflow as tf
from sklearn import datasets
from sklearn.cluster import *

from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import numpy as np

from parts import *

iris = datasets.load_iris()
digits = datasets.load_digits()

# X = iris.data

X = digits.data.reshape((1797, 8, 8, 1))
X_ = digits.data

targets = np.zeros((1797, 10))
for i in range(1797):
    targets[i, digits.target[i]] = 1

dataset = {'x': X, 'y': np.zeros((1797, 2)), 'label': digits.target}

COLORS = [mcolors.cnames['black'],
          mcolors.cnames['red'],
          mcolors.cnames['green'],
          mcolors.cnames['blue'],
          mcolors.cnames['sienna'],
          mcolors.cnames['silver'],
          mcolors.cnames['olive'],
          mcolors.cnames['fuchsia'],
          mcolors.cnames['darkviolet'],
          mcolors.cnames['lime']]

LAYERS = [Convolutional((4, 4), 10, name='conv1'),
          Convolutional((2, 2), 20, name='conv2'),
          Flatten(name='flat'),
          FullConnection(30, name='fc1'),
          FullConnection(2, name='fc2', activation_function=None)
          ]

NUM_MIX_COMPONENTS = 10
REPRESENTATION_DIMENSIONALITY = 2


class CNN:
    def __init__(self, input_shape, layers, sess, **kwargs):
        """
        """

        self.num_mixture_components = kwargs.get('num_mixture_components', NUM_MIX_COMPONENTS)

        self.sess = sess

        self.input = tf.placeholder(tf.float32, (None,) + input_shape + (1,))

        current_layer = self.input

        for layer in layers:
            current_layer, w, b = layer.build(current_layer)

        self.out = current_layer

        self.tgt = tf.placeholder(tf.float32, [None, 2])
        self.loss = tf.reduce_mean(tf.square(self.out - self.tgt))

        self.train_op = tf.train.RMSPropOptimizer(0.01, momentum=0.9).minimize(self.loss)

    def train(self, _x, _y):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input: _x, self.tgt: _y})
        return loss

    def fwd(self, _x):
        return self.sess.run(self.out, feed_dict={self.input: _x})


class GMM_NLL:
    """
    """

    def __init__(self, input_tensor, K, dim, sess):
        """
        """

        self.input = input_tensor
        self.sess = sess

        self.K = K
        self.dim = dim

        # Definition of GMM components
        self.means = tf.placeholder(tf.float32, (self.K, self.dim))
        self.covariances = tf.placeholder(tf.float32, (self.K, self.dim, self.dim))
        self.weights = tf.placeholder(tf.float32, (self.K))

        cov_inv = tf.matrix_inverse(self.covariances)
        cov_det = tf.matrix_determinant(self.covariances)

        self.NLL = None


class NN:
    def __init__(self, sess):
        self.sess = sess

        # Make the neural network
        self.input = tf.placeholder(tf.float32, [None, 4])

        W1 = tf.Variable(tf.random_normal([4, 6], stddev=0.1))
        b1 = tf.Variable(tf.zeros([6]))
        h = tf.nn.sigmoid(tf.matmul(self.input, W1) + b1)
        W2 = tf.Variable(tf.random_normal([6, 2], stddev=0.1))
        b2 = tf.Variable(tf.zeros([2]))

        self.out = tf.matmul(h, W2) + b2
        self.tgt = tf.placeholder(tf.float32, [None, 2])
        self.loss = tf.reduce_mean(tf.square(self.out - self.tgt))
        self.train_op = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

    def train(self, _x, _y):
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.input: _x, self.tgt: _y})
        return loss

    def fwd(self, _x):
        return self.sess.run(self.out, feed_dict={self.input: _x})


class Display:
    def __init__(self, max_clusters=10):

        plt.ion()

        self.figs = [None, None]

        self.max_clusters = 10
        self.figs[0] = plt.subplot(121)
        self.plots = [[], []]
        for i in range(10):
            self.plots[0].append(plt.plot([], [], linewidth=0, marker='o', color=COLORS[i])[0])
        self.figs[1] = plt.subplot(122)
        for i in range(10):
            self.plots[1].append(plt.plot([], [], linewidth=0, marker='o', color=COLORS[i])[0])

        self.xlim = [-5, 5]
        self.ylim = [-5, 5]

        self.figs[0].set_xlim(self.xlim)
        self.figs[0].set_ylim(self.ylim)
        self.figs[1].set_xlim(self.xlim)
        self.figs[1].set_ylim(self.ylim)

    def update(self, data, fig_num=0):

        xmins = [np.min(d[:, 0]) for d in data]
        xmaxs = [np.max(d[:, 0]) for d in data]
        ymins = [np.min(d[:, 1]) for d in data]
        ymaxs = [np.max(d[:, 1]) for d in data]

        xmin = min(xmins)
        xmax = max(xmaxs)
        ymin = min(ymins)
        ymax = max(ymaxs)

        xmin = xmin - 0.05 * (xmax - xmin)
        xmax = xmax + 0.05 * (xmax - xmin)
        ymin = ymin - 0.05 * (ymax - ymin)
        ymax = ymax + 0.05 * (ymax - ymin)

        for i in range(len(data)):
            x_dat = data[i][:, 0]
            y_dat = data[i][:, 1]

            self.plots[fig_num][i].set_data(x_dat, y_dat)
            self.figs[fig_num].set_xlim([xmin, xmax])
            self.figs[fig_num].set_ylim([ymin, ymax])

        plt.pause(0.05)


class GaussianCluster:
    """
    """

    def __init__(self, mean, covariance, weight):
        """
        """

        self.mean = mean
        self.covariance = covariance
        self.weight = weight

        self.datapoints = []

        self.displacement = np.zeros(self.mean.shape)

    def add_datapoint(self, datapoint):
        """
        """

        self.datapoints.append(datapoint)

    def calculate_displacement(self, clusters):
        """
        """

        self.displacement = np.zeros(self.mean.shape)

        for c in clusters:
            if c != self:
                delta = self.mean - c.mean

                delta_mag = np.dot(delta, delta.T)

                fr = 1.0 / delta_mag

                self.displacement += (delta / delta_mag) * fr

        if np.dot(self.displacement, self.displacement.T) > 1.0:
            self.displacement /= np.dot(self.displacement, self.displacement.T)


def cluster(data, num_clusters=10):
    """
    Perform clustering on y
    """

    y = data['y']

    clusters = []

    gmm = GaussianMixture(num_clusters)
    gmm.fit(y)

    weights = gmm.weights_
    means = gmm.means_
    covariances = gmm.covariances_

    probs = gmm.predict_proba(y)

    for i in range(num_clusters):
        clusters.append(GaussianCluster(means[i], covariances[i], weights[i]))

    # Assign datapoints to each cluster
    cluster_ids = np.argmax(probs, axis=1)

    for i in range(y.shape[0]):
        clusters[cluster_ids[i]].add_datapoint((data['x'][i], data['y'][i], data['label'][i]))

    return clusters


def plot_clusters(clusters, num_labels=10):
    """
    """

    plt_pts = [[d[1] for d in c.datapoints] for c in clusters]
    plt_pts = [np.array(c) for c in plt_pts]

    plt_pts_by_label = []
    for i in range(num_labels):
        plt_pts_by_label.append([])

    for c in clusters:
        for d in c.datapoints:
            plt_pts_by_label[d[2]].append(d[1])

    plt_pts_by_label = [np.array(p) for p in plt_pts_by_label]

    display.update(plt_pts, 0)
    display.update(plt_pts_by_label, 1)


def BhattacharyyaDistance(dist1, dist2):  # mean1, cov1, mean2, cov2):
    """
    Calculate the BhattacharyyaDistance between two distirbutions
    """

    mean1, cov1 = dist1.mean, dist1.covariance
    mean2, cov2 = dist2.mean, dist2.covariance

    mean_diff = mean1 - mean2
    cov = (cov1 + cov2) / 2
    cov_inv = np.linalg.inv(cov)
    dB1 = np.dot(mean_diff, np.dot(cov_inv, mean_diff)) / 8
    dB2 = 0.5 * np.log(np.linalg.det(cov) / np.sqrt(np.linalg.det(cov1) * np.linalg.det(cov2)))

    return dB1 + dB2


def HellingerDistance(dist1, dist2):  # mean1, cov1, mean2, cov2):
    """
    """

    mean1, cov1 = dist1.mean, dist1.covariance
    mean2, cov2 = dist2.mean, dist2.covariance

    mean_diff = mean1 - mean2
    cov = (cov1 + cov2) / 2

    cov_det = np.linalg.det(cov)
    det1 = np.linalg.det(cov1)
    det2 = np.linalg.det(cov2)

    tmp1 = (np.power(det1, 0.25) * np.power(det2, 0.25)) / np.sqrt(cov_det)
    cov_inv = np.linalg.inv(cov)
    tmp2 = np.dot(mean_diff, np.dot(cov_inv, mean_diff)) / 8

    dist_sq = 1.0 - tmp1 * np.exp(-tmp2)

    return np.sqrt(dist_sq)


def extract_training_data(clusters):
    """
    """

    X = []
    y = []

    for c in clusters:
        for d in c.datapoints:
            X.append(d[0])
            y.append(d[1] + c.displacement - 0.05 * (d[1] - c.mean))

    X = np.array(X)
    y = np.array(y)

    return X, y


def step(dataset, nn):
    # 1.  Do a forward pass
    dataset['y'] = nn.fwd(dataset['x'])

    # 2. Perform clustering
    clusters = cluster(dataset)

    # 3. Calculate the distances between the distributions
    distances = calc_distances(clusters)

    # 4. Displace the center of each cluster using force directed graph
    for c in clusters:
        c.calculate_displacement(clusters)

    # 5. Set the target position of each item in the cluster to the mean of the cluster
    x, y = extract_training_data(clusters)

    # 6. Train the CNN\
    for j in range(100):
        nn.train(x, y)

    # 7. Plot the clusters
    plot_clusters(clusters)

    return clusters


def calc_distances(data, dist_func=HellingerDistance):
    """
    """

    distances = np.zeros((len(data), len(data)))

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distances[i, j] = dist_func(data[i], data[j])
            distances[j, i] = dist_func(data[j], data[i])

    return distances


def get_displacement(data, distances):
    """
    Determine the centers of the data after moving
    """

    displacements = np.zeros((len(data), 2))

    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                delta = data[i] - data[j]
                delta_mag = np.dot(delta, delta.T)
                if delta_mag < 1.0:
                    displacement = (delta / delta_mag) * 0.001 / delta_mag
                    displacements[i, 0] += displacement[0, 0]
                    displacements[i, 1] += displacement[0, 1]
    return displacements


sess = tf.InteractiveSession()
cnn = CNN((8, 8), LAYERS, sess)
sess.run(tf.global_variables_initializer())

display = Display()
