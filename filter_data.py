from scipy.io import loadmat, savemat
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y



def load(indices, cutoff):
    prefered = loadmat("../prefered.mat")
    prefered = prefered['prefered']
    preData = prefered[0]

    # y = prefered[0][0]
    # print(y.shape)

    X_train = []

    for i in range(151):
        if i in indices:   
            Sig = preData[i][0]

            final = np.empty((Sig.shape[0], 16001))
            pr = []

            for j in range(Sig.shape[0]):
                    pr.append(np.mean(Sig[j][0], axis=0))

            for k in range(final.shape[0]):
                    final[k] = pr[k]
            
            ff = np.mean(final, axis=0)
            ff = ff[2400:3500]
            
            # Filter requirements.
            order = 6
            fs = 5000.0       # sample rate, Hz

            ff = butter_lowpass_filter(ff, cutoff, fs, order)
            X_train.append(ff)
            # plt.plot(ff)
            # plt.show()

    X_train = np.array(X_train)
    return X_train


def filter(X, cutoff, order):
    fs = 5000.0       # sample rate, Hz
    X_train = []

    for x in X:
        ff = butter_lowpass_filter(x, cutoff, fs, order)
        X_train.append(ff)
        
    return np.array(X_train)
# load()