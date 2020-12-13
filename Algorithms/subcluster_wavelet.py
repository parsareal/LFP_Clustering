import pywt
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn import preprocessing

import sys
import algorithm
sys.path.insert(1, '../')
import loadData
import plots
import filter_data

startScale = 50
endScale = 120
wavelet = 'morl'

def subCluster(cluster):

    y_pred, X_train = algorithm.cluster()
    y, dates = loadData.load()

            # SELECT CERTAIN CLUSTER
    array = []
    arrDates = []
    for i in range(151):
        if y_pred[i] == cluster:
            array.append(i)
            arrDates.append(dates[i])
    # X = np.array(array)
    D = np.array(arrDates)
    
    X = []
    for xx in X_train[y_pred==cluster]:
        X.append(xx)
    X = np.array(X)
    X = preprocessing.scale(X, axis=1)

    cutoff = 100.0  # desired cutoff frequency of the filter, Hz
    # filtered_x = filter_data.load(array, cutoff)

            # CAL WAVELET OF THE SIGNALS
    array = []
    for i in range(X.shape[0]):
        widths = np.arange(startScale, endScale)
        cwtmatr, freqs = pywt.cwt(X[i], widths, wavelet)
        array.append(np.mean(cwtmatr, axis=0))
        # if D[i] == u'130207':
        #     plt.plot(np.mean(cwtmatr, axis=0))
        #     plt.show()

    X = np.array(array)
    X = preprocessing.scale(X, axis=1)

            # FLATTEN DATA
    array = []
    for x in X:
        x = x.flatten()
        array.append(x)
    X = np.array(array)

    clustering = KMeans(n_clusters=2, random_state=0).fit(X)
    pred = clustering.labels_
    print(pred)

            # GET ORIGINAL DATA BACK
    array = []
    for i in range(151):
        if y_pred[i] == cluster:
            array.append(X_train[i])
    X = np.array(array)
    X = filter_data.filter(X, 40, 6)
    X = preprocessing.scale(X, axis=1)

    dt = 0.0002  # 5000 Hz sampling
    frequencies = pywt.scale2frequency(wavelet, [startScale, endScale]) / dt
    print(frequencies)

    append_title = "DBSCAN(first classifier) & KMeans(second classifier) Algorithm \n Lowpass filter for wavelet = "+ str(cutoff) +"Hz \n" + wavelet + " wavelet Freqs = [{}, {}])".format(str("%.3f" % round(frequencies[1],2)), str("%.3f" % round(frequencies[0],2))) + " \n wavelet Scales = [{}, {}])".format(startScale, endScale)

    plots.plotCluster(0, pred, X, D, 1, 1500, 400, append_title)
    # plots.plotCluster(1, pred, X, D, 1, 1500, 400, append_title)


subCluster(1)

# for i in range(68):
#     if D[i][0] == u'120814':
#         widths = np.arange(1, 30)
#         cwtmatr, freqs = pywt.cwt(X[i], widths, 'mexh')
#         plt.imshow(cwtmatr, aspect='auto')  # doctest: +SKIP
#         # plt.ylabel('Scale')
#         # plt.xlabel('Time')
#         # plt.plot(np.mean(cwtmatr, axis=0))
#         # plt.plot(X_train[0])

# plt.show()