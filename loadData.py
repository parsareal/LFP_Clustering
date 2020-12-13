from scipy.io import loadmat, savemat
import numpy as np

def load():
    data = loadmat("../data/LFP_OCP.mat")
    dates = loadmat("../data/dates.mat")

    data = data['LFP_OCP']
    dates = dates['dates']

    y = data[0]
    dates = dates[0]

    return y, dates

# load()

