import numpy as np
import matplotlib.pyplot as plt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from scipy.io import loadmat,savemat
from scipy import signal
import copy
import sys

sys.path.insert(1, '/Analysis')
import analysisC1
import analysisC0


#               INITIALIZE PARAMETERS

seed = 0
np.random.seed(seed)
data = loadmat("LFP_Avg.mat")
data1 = loadmat("LFP_OCP.mat")
s = data1['LFP_OCP']
# s = data['ALL_LFP_Or']
# x = s[0]

y = data['LFP_Avg']

# y = x[0][3][0]
# y = y[0:16000]
# plt.plot(y)
# plt.show()

sampling_rate = 5
start = 400
end = 1500
clusterNum = 3

# y = y[0::16000]

# y = s[0][0]
date = s[0][1]


#               CUSTOMIZE TRAIN DATA
array = []
for i in range(151):
    x = y[0][i][0]
    x = x[2000:4000]
    x = x[start:end:sampling_rate]
    array.append(x)
X_train = np.array(array)

X_train = TimeSeriesScalerMeanVariance().fit_transform(X_train[:151])



#               Euclidean k-means
km = TimeSeriesKMeans(n_clusters=clusterNum, verbose=True, random_state=seed)
pred = km.fit_predict(X_train)
y_pred = copy.copy(pred)
print(y_pred)


#               START SUBCLASING

# clss = []
# for xx in X_train:
#     xx1 = xx.flatten()
#     fs = float(220)
#     fc = float(10)
#     w = float(fc / (fs / 2))
#     b, a = signal.butter(5, w, 'low')
#     xx1 = signal.filtfilt(b, a, xx1)
#     clss.append(xx1.tolist())
    
# X_train = np.array(clss)

# cls0 = []
# for xx in X_train[y_pred == 0]:
#     cls0.append(xx.flatten().tolist())
# cls0 = np.array(cls0)

# km = TimeSeriesKMeans(n_clusters=clusterNum, verbose=True, random_state=seed)
# pred = km.fit_predict(cls0)
# y_pred = copy.copy(pred)
# print(y_pred)



def writeLabels():
    with open('labels.txt', 'w') as f:
        for item in y_pred:
            f.write("%s\n" % item)

# writeLabels()


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
        ind = np.where(X_train == xx)
        ind = ind[0][0]
        sigDate = date[ind][0][0]
        plt.text(220, 2, str(sigDate), fontsize=8)
        
        #       SHOW STIMULES ON PLOT
        plt.axvline(x=(100/sampling_rate), color='red')
        plt.axis('off')

        #       ANALYSE THE CLUSTER
        # properties, xx1, onsetFirstSDTres, onsetSecondSDTres, maxFirstSDTres, maxSecondSDTres, fc = analysisC1.analyseCluster1(xx, properties, sampling_rate, sigDate, counters)
        # properties, xx1, onsetFirstSDTres, onsetSecondSDTres, maxFirstSDTres, maxSecondSDTres, fc = analysisC0.analyseCluster0(xx, properties, sampling_rate, sigDate, counters)

        plt.plot(xx)


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

    plt.suptitle("Sampling rate = "+ str(sampling_rate) + ", Period = "+ str(start) + ":" + str(end) + ", Cluster = 0-" + str(yi))
    # plt.suptitle("Sampling rate = "+ str(sampling_rate) + ", Period = "+ str(start) + ":" + str(end) + ", Cluster = " + str(yi) + 
    #         "\n Lowpass filter " + str(fc) + "Hz (Just for onset and pick detecting)" + 
    #             "\n onsetFirstSDTres = " + str(onsetFirstSDTres) + ", onsetSecondSDTres = " + str(onsetSecondSDTres) +
    #                 "\n pickFirstSDTres = " + str(maxFirstSDTres) + 
    #                     "\n start for pick detecting =  120ms" + 
    #                         "\n final window size for pick detecting =  40ms")
                        # "\n Onset Manipulated signals = 130204, 050831, 081009, 120816, 070712" +
                        # "\n Pick Manipulated signals = 060814, 070418, 070926, 070412, 061110, 061114")
    
    plt.show()



    #        PLOT HISTOGRAM AND PDF

    # analyseC0(clusterMems, properties)
    # analyseC1(clusterMems, properties)



def analyseC0(clusterMems, properties):
    analysisC0.writeExcel(clusterMems, properties)
    analysisC0.dipTestDist(properties, clusterMems)
    analysisC0.analyseDistance(properties, clusterMems)
    # analysisC0.analyseOnset(properties, clusterMems)
    # analysisC0.analysePick(properties, clusterMems)

def analyseC1(clusterMems, properties):
    analysisC1.writeExcel(clusterMems, properties)
    analysisC1.dipTestDist(properties, clusterMems)
    # analysis.analyseDistance(properties, clusterMems)
    analysisC1.analyseOnset(properties, clusterMems)
    # analysis.analysePick(properties, clusterMems)


def plotSignals(filter_stat, index):
    counters = 0

    for yi in range(clusterNum):
        plt.subplot(15, clusterNum, 1 + yi)
        # plt.plot(X_train[0], "r-")
        plt.axis('off')
        plt.plot(km.cluster_centers_[yi].ravel(), "r-")
        # plt.plot(X_train[yi], "r-")
        for xx in X_train[y_pred == yi]:
            counters = counters + 1
            plt.plot(xx.ravel(), "k-", alpha=.2)


        plt.title("Cluster " + str(yi) + " = " + str(counters))
        counters = 0

        for xx in X_train[y_pred == yi]:
            counters = counters + 1
            if counters < (13*index):
                continue
            if counters == (13*(index+1)):
                break
            plt.subplot(15, clusterNum, yi + 1 + clusterNum*(counters - 13*index + 1))
            plt.axvline(x=100/sampling_rate, color='red')
            plt.axis('off')
            ind = np.where(X_train == xx)
            ind = ind[0][0]
            sigDate = date[ind][0][0]
            plt.text(150, 2, str(sigDate))
            plt.plot(xx)
        

        # plt.subplot(15, clusterNum, yi + 1 + clusterNum*14)
        # plt.axis(xmin=start, xmax=end)
        
        plt.subplot(15, clusterNum, yi + 1 + clusterNum*14)
        plt.axis(xmin=-0.22, xmax=1.5)
        plt.axvline(x=0, color='red')    


        counters = 0

    # plt.savefig('out.pdf')

    plt.suptitle("Sampling rate = "+ str(sampling_rate) + ", Period = "+ str(start) + ":" + str(end) + "\n Savitzky_Golay_Filter = " + str(filter_On) + "\n Plot = " + str(index+1))
    plt.show()



def manipulate():
    #### manipulate results ####

    counter = 0
    index = 0
    indexes = [46, 49, 50]
    for i in pred:
        found = False
        if i == 0:
            counter = counter + 1
            found = True
        # if counter in indexes:
        #     y_pred[index] = 1
        
        if (counter in indexes) & found:
            y_pred[index] = 1
        
        if (counter == 30) & found:
            y_pred[index] = 2

        index = index + 1


    counter = 0
    index = 0
    indexes = [1, 4, 5, 6, 7, 8, 9]
    for i in pred:
        found = False
        if i == 1:
            counter = counter + 1
            found = True

        if found:
            y_pred[index] = 2
        if (counter in indexes) & found:
            y_pred[index] = 1

        index = index + 1


    counter = 0
    index = 0
    for i in pred:
        found = False
        if i == 3:
            counter = counter + 1
            found = True

        if found:
            y_pred[index] = 2
        if (counter == 1) & found:
            y_pred[index] = 3

        index = index + 1


    counter = 0
    index = 0
    indexes = [5, 6, 7, 8, 9, 25]
    for i in pred:
        found = False
        if i == 4:
            found = True
            counter = counter + 1

        if found:
            y_pred[index] = 1
        if (counter in indexes) & found:
            y_pred[index] = 3

        index = index + 1



# manipulate()
plotCluster(0)
plotCluster(1)
plotCluster(2)
# plotCluster(3)
# plotCluster(4)


# plotSignals(False, 0)
# plotSignals(False, 1)
# plotSignals(False, 2)
# plotSignals(False, 3)
# plotSignals(False, 5)

