from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering, Birch, OPTICS
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from scipy.io import loadmat, savemat
from scipy import signal
import copy

import sys

sys.path.insert(1, '../')
import loadData
import plots
import filter_data


def cluster():
    #   INITIALIZE PARAMETERS
    y, dates = loadData.load()

    sampling_rate = 1
    start = 400
    end = 1500
    eps = 26
    min_samples = 25

    threshold = 0.9
    branching_factor = 5

    #   CUSTOMIZE TRAIN DATA
    array = []
    for i in range(151):
        # x = y[i][0]
        x = y[0][i]
        # x = x[2000:4000]
        x = x[start:end:sampling_rate]
        array.append(x)

    X_train = np.array(array)
    X_train = preprocessing.scale(X_train, axis=1)

    array = []
    for x in X_train:
        x = x.flatten()

        fs = float(end - start)
        fc = float(20)
        w = float(fc / (fs / 2))

        b, a = signal.butter(5, w, 'low')
        xx1 = signal.filtfilt(b, a, x)

        array.append(xx1)

    X_train = np.array(array)

    print(np.linalg.norm(X_train[15] - X_train[14]))

    # nbrs = NearestNeighbors(n_neighbors=len(X_train)).fit(X_train)
    # distances, indices = nbrs.kneighbors(X_train)
    # # print(distances.flatten().shape)
    # # print(distances.shape)
    # sortedDiss = distances.flatten()
    # sortedDiss = np.sort(sortedDiss)
    # index = np.arange(0,22801)
    # plt.plot(sortedDiss, index)
    # plt.ylabel('indices')
    # plt.xlabel('distance')
    # plt.show()

    # clustering = OPTICS(min_samples=5).fit(X_train)
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_train)
    # clustering = KMeans(n_clusters=3, random_state=0).fit(X_train)
    # clustering = AgglomerativeClustering(n_clusters=3).fit(X_train)
    # clustering = Birch(branching_factor=branching_factor, n_clusters=3, threshold=threshold, compute_labels=True).fit(X_train)

    for n, i in enumerate(clustering.labels_):
        if i == -1:
            clustering.labels_[n] = 2
    y_pred = clustering.labels_

    sink_indices = [u'040928', u'050606', u'061211']
    source_indices = [u'071004']
    for i in range(151):
        if dates[i][0] in sink_indices:
            y_pred[i] = 1
        if dates[i][0] in source_indices:
            y_pred[i] = 0

    print(type(y_pred))

    append_title = "DBSCAN Algorithm with minSamples={}, eps={}".format(min_samples, eps)

    # fcluster2 = filter_data.filter(X_train, 50, 6)

    plots.plotCluster(0, y_pred, X_train, dates, sampling_rate, end, start, append_title, analyse=True)
    # plots.plotCluster(1, y_pred, X_train, dates, sampling_rate, end, start, append_title, analyse=True)
    # plots.plotCluster(2, y_pred, X_train, dates, sampling_rate, end, start, append_title)
    # plots.plotCluster(2, y_pred, fcluster2, dates, sampling_rate, end, start, append_title)

    return y_pred, X_train


def write_labels(y_pred):
    with open('../data/labels.txt', 'w') as f:
        for item in y_pred:
            f.write("%s\n" % item)


cluster()
