from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.stats import vonmises
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.signal import hilbert, chirp


sys.path.insert(1, '../')

sig_date = '080627'
sig_type = 'source'


def load_spikes():
    data = loadmat("../data/V1DATA.mat")
    data = data['V1DATA']
    dates = data['date'][0]
    spikes = data['spikes'][0]
    task_name = data['task_name'][0]
    indices = [i for i, x in enumerate(dates) if x == sig_date]
    new_indices = []
    for i in indices:
        if task_name[i] == 'Orientation':
            new_indices.append(i)
    spikes_new = []
    for i in new_indices:
        spikes_new.append(spikes[i])
    spikes_new = np.array(spikes_new)
    return spikes_new
    # print(spikes_new.shape)


def find_amps(data):
    # data = loadmat("../data/spikes/" + sig_type + "/" + sig_date + ".mat")
    # # data = loadmat("../data/spikes/" + sig_date + ".mat")
    # data = data['V1DATA1']
    # data = data[0]

    # orientations = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 357]
    rates_d0 = []
    rates_d1 = []

    for i in range(8):
        or_d0 = np.mean(data[i], axis=0)
        or_d1 = np.mean(data[i+8], axis=0)

        or_d0 = or_d0[500:]
        or_d1 = or_d1[500:]

        spike_amp_d0 = len(np.nonzero(or_d0)[0])
        spike_amp_d1 = len(np.nonzero(or_d1)[0])

        rates_d0.append(spike_amp_d0)
        rates_d1.append(spike_amp_d1)

    rates_d0.extend(rates_d1)
    return rates_d0


def find_params(data, orientations):
    data = np.array(data)
    
    first_or_ind = np.argmax(data)
    first_or = orientations[first_or_ind]
    if first_or < 180:
        second_or = first_or + 180
        second_or_ind = orientations.index(second_or)
    
    if first_or >= 180:
        second_or = first_or - 180
        second_or_ind = orientations.index(second_or)
    
    first_amp = data[first_or_ind]
    second_amp = data[second_or_ind]
    
    if first_or >= 180:
        params = [second_amp, 1/(10/180*np.pi), np.pi/2, first_amp, 1/(40/180*np.pi), np.pi*3/2]
    else:
        params = [first_amp, 1/(40/180*np.pi), np.pi/2, second_amp, 1/(10/180*np.pi), np.pi*3/2]
    
    return params


def van_mos(x, a, b, c, d, e, f):
    return a * np.exp(b * (np.cos(x - c) - 1)) + d * \
        np.exp(e * (np.cos(x - f) - 1))


def plot_curve(data, orientations):
    orientations = np.array(orientations)
    ororientations_rad = orientations / 180 * np.pi

    ors = np.linspace(0, 360, num=1000)
    ors_rad = ors / 180 * np.pi

    # data = data / np.max(data)
    param, param_cov = curve_fit(van_mos, ororientations_rad, data)
    a = param[0]
    b = param[1]
    c = param[2]
    d = param[3]
    e = param[4]
    f = param[5]

    # y = params[0] * np.exp(params[1] * (np.cos(ors_rad - params[2]) - 1)) + params[3] * \
    #     np.exp(params[4] * (np.cos(ors_rad - params[5]) - 1))
    # y = a * np.exp(b * (np.cos(ors_rad - c) - 1)) + d * \
    #     np.exp(e * (np.cos(ors_rad - c - np.pi) - 1))
    y = a * np.exp(b * (np.cos(ors_rad - c) - 1)) + d * \
            np.exp(e * (np.cos(ors_rad - f) - 1))

    # fig, ax = plt.subplots()
    # ax.scatter(orientations, y1, color='green', label='Ratio per orientation')

    # for i, txt in enumerate(y1):
    #     ax.annotate("%.2f" % round(txt, 2), (orientations[i], y1[i]))
    # ax.legend()
    plt.plot(ors, y, label='von Mises fit')
    return param


def cal_error(orientations, y_data, params):
    orientations = np.array(orientations)
    ororientations_rad = orientations / 180 * np.pi

    a = params[0]
    b = params[1]
    c = params[2]
    d = params[3]
    e = params[4]
    f = param[5]

    # y1 = a * np.exp(b * (np.cos(ororientations_rad - c) - 1)) + d * \
    #     np.exp(e * (np.cos(ororientations_rad - c - np.pi) - 1))
    y1 = a * np.exp(b * (np.cos(ororientations_rad - c) - 1)) + d * \
        np.exp(e * (np.cos(ororientations_rad - f) - 1))

    # err = mean_squared_error(y_data, y1)
    err = 0
    sus = 0
    mean = np.mean(y_data)
    for i in range(len(y_data)):
        err += (y_data[i] - y1[i]) ** 2

    for yy in y_data:
        sus += (yy - mean) ** 2
    # sus = mean_squared_error(y_data, np.mean(y_data))
    # acc = 1 - err / (np.mean(y_data) ** 2)
    acc = 1 - (err / sus)
    return acc


if __name__ == '__main__':
    spikes = load_spikes()
    source_data = find_amps(spikes)
    source_orientations = [0, 22, 45, 67, 90, 112, 135, 157, 180, 202, 225, 247, 270, 292, 315, 357]
    # params = find_params(source_data, source_orientations)
    param = plot_curve(source_data, source_orientations)
    acc = cal_error(source_orientations, source_data, param)
    acc = "%.2f" % round(acc, 2)
    plt.scatter(source_orientations, source_data, color='red', label='Data Amps')
    plt.legend()
    plt.title('Orientation Tuning Signal: ' + sig_date + "\n R1 = {}".format(acc))
    plt.xlabel('orientation')
    plt.savefig('../data/spikes/' + sig_type + '/' + sig_date + '.png')
    plt.show()
