import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import numpy as np
from scipy import signal
import pywt
import umap
from sklearn.cluster import DBSCAN, KMeans
import copy
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler


import sys
sys.path.insert(1, '../')
import loadData


y, dates = loadData.load()

eps = 24.5
min_samples = 27
sampling_rate = 1.5
start = 400
end = 1500

array = []
for i in range(151):
    x = y[i][0]
    x = x[2000:4000]
    x = x[start:end:1]
    array.append(x)
X_train = np.array(array)

X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:151])

array = []
for x in X_train:
    x = x.flatten()
    array.append(x)
X_train = np.array(array)


print(pywt.wavelist(kind='continuous'))

# array = []
# for i in range(151):
#     coeffs = pywt.wavedec(X_train[i], 'db1', level=6)
#     array.append(coeffs[0])
#     # widths = np.arange(1, 31)
#     # cwtmatr, freqs = pywt.cwt(X_train[i], widths, 'mexh')
#     # array.append(np.mean(cwtmatr, axis=0))


# filtered = np.array(array)




clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X_train)
for n, i in enumerate(clustering.labels_):
    if i == -1:
        clustering.labels_[n] = 2

pred = clustering.labels_
y_pred = copy.copy(pred)
print(pred)

# array = []
# for xx in X_train[y_pred == 1]:
#     array.append(xx)
# array = np.array(array)



def plotCluster(yi):
    clusterNum = 7
    plt.subplot(15, clusterNum, 1)
    plt.axis('off')
    counters = 0

    y_test = y_pred.tolist()
    clusterMems = y_test.count(yi)
    properties = [[0 for x in range(4)] for y in range(clusterMems)] 
    print(clusterMems)

    for xx in X_train[y_pred == yi]:
        counters = counters + 1
        if counters <= 13:
            plt.subplot(15, clusterNum, 1 + clusterNum*counters)
        if 13 < counters <= 26:
            plt.subplot(15, clusterNum, 2 + clusterNum*(counters - 13))
        if 26 < counters <= 39:
            plt.subplot(15, clusterNum, 3 + clusterNum*(counters - 26))
        if 39 < counters <= 52:
            plt.subplot(15, clusterNum, 4 + clusterNum*(counters - 39))
        if 52 < counters <= 65:
            plt.subplot(15, clusterNum, 5 + clusterNum*(counters - 52))
        if 65 < counters <= 78:
            plt.subplot(15, clusterNum, 6 + clusterNum*(counters - 65)) 
        if counters > 78:
            plt.subplot(15, clusterNum, 7 + clusterNum*(counters - 78)) 


        #       IDENTIFY WITH EACH SIGNAL DATE
        
        #       SHOW STIMULES ON PLOT
        plt.axvline(x=(100/sampling_rate), color='red')
        plt.axis('off')

        ind = np.where(X_train == xx)
        ind = ind[0][0]
        sigDate = dates[ind][0]
        plt.text((end-start), 1, str(sigDate), fontsize=8)
        
        #       ANALYSE THE CLUSTER
        # fs = float(220)
        # fc = float(10)
        # w = float(fc / (fs / 2))

        # b, a = signal.butter(5, w, 'low')
        # xx1 = signal.filtfilt(b, a, xx)

        widths = np.arange(29, 30)
        cwtmatr, freqs = pywt.cwt(xx, widths, 'mexh')
        plt.plot(cwtmatr[0])


    counters = 0

    plt.subplot(15, clusterNum, 1 + clusterNum*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')    

    plt.subplot(15, clusterNum, 2 + clusterNum*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')     
    
    plt.subplot(15, clusterNum, 3 + clusterNum*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')
    
    plt.subplot(15, clusterNum, 4 + clusterNum*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')     
    
    plt.subplot(15, clusterNum, 5 + clusterNum*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')

    plt.subplot(15, clusterNum, 6 + clusterNum*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')

    plt.subplot(15, clusterNum, 7 + clusterNum*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')  

    # plt.suptitle("AgglomerativeClustering(linkage) Algorithm \n Sampling rate = "+ str(sampling_rate) + ", Period = "+ str(start) + ":" + str(end) + ", Cluster = " + str(yi) +
    #     # "\n eps = " + str(eps) + ", min_samples = " + str(min_samples) + 
    #         "\n clusterMembers = " + str(clusterMems))
    
    plt.show()



# plotCluster(0)
# plotCluster(1)
# plotCluster(2)

startScale = 80
endScale = 120
dt = 0.0002  # 100 Hz sampling
frequencies = pywt.scale2frequency('morl', [startScale, endScale]) / dt
print(frequencies)

for i in range(151):
    if dates[i][0] == u'100217':
        widths = np.arange(startScale, endScale)
        cwtmatr, freqs = pywt.cwt(X_train[i], widths, 'morl')
        print(cwtmatr.shape)
        print(freqs.shape)
        print(freqs)
        # plt.imshow(cwtmatr, aspect='auto')  # doctest: +SKIP
        plt.ylabel('Scale')
        plt.xlabel('Time')
        plt.plot(np.mean(cwtmatr, axis=0))
        # plt.plot(X_train[0])

plt.show() # doctest: +SKIP