import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy import signal
from scipy.stats import norm, uniform, gaussian_kde
from unidip import dip

from xlwt import Workbook
import filter_data
import plots


def dip_test(properties, cluster_members, feature=3):
    data = []
    outliers = [u'121010', u'091110', u'091109', u'091106']

    for ii in range(cluster_members):
        if not properties[ii][0] in outliers:
            data.append(float(properties[ii][feature]))

    data = np.array(data)
    data = np.msort(data)
    intervals = dip.diptst(data)

    t_range = np.linspace(0, 0.2, 200)
    kde = gaussian_kde(data)
    plt.plot(t_range, kde(t_range) / 100)
    plt.grid()
    plt.title('Distance PDF' + '\n p_value = ' + str(intervals[1]))

    plt.show()


def analyse_distance(properties, cluster_members, feature=3):
    data = []
    outliers = [u'121010', u'091110', u'091109', u'091106']

    for ii in range(cluster_members):
        if not properties[ii][0] in outliers:
            data.append(float(properties[ii][feature]))

    num_bins = 13
    plt.ylabel('Cluster 0 signals')
    plt.xlabel('Cluster 0 onsets')
    plt.title('Distance Histogram Plot \n Bins = ' + str(num_bins) + '\n Cluster 0')
    plt.grid()
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)

    # mu = np.mean(data)
    # sigma = statistics.stdev(data)
    # print(mu)
    # print(sigma)
    # y = mlab.normpdf(bins, mu, sigma)
    # plt.plot(bins, y, 'r--')
    
    plt.show()


def analyseOnset(properties, cluster_members):
    data = []
    for ii in range(cluster_members):
        data.append(float(properties[ii][1]))

    num_bins = 8
    plt.ylabel('Cluster 0 signals')
    plt.xlabel('Cluster 0 distances')    
    plt.title('Onset Histogram Plot \n Bins = ' + str(num_bins) + '\n Cluster 0')
    plt.grid()
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)

    # print(bins)
    # mu = np.mean(data)
    # sigma = statistics.stdev(data)
    # y = mlab.normpdf(bins, mu, sigma)
    # plt.plot(bins, y, 'r--')
    
    plt.show()


def analysePick(properties, cluster_members):
    data = []
    for ii in range(cluster_members):
        data.append(float(properties[ii][2]))

    num_bins = 8
    plt.ylabel('Cluster 0 signals')
    plt.xlabel('Cluster 0 distances')    
    plt.title('Pick Histogram Plot \n Bins = ' + str(num_bins) + '\n Cluster 0')
    plt.grid()
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)

    # mu = np.mean(data)
    # sigma = statistics.stdev(data)
    # y = mlab.normpdf(bins, mu, sigma)
    # plt.plot(bins, y, 'r--')
    
    plt.show()


def analyse_cluster0(xx, properties, sampling_rate, sig_date, counters):
    
    #       LOW_PASS FILTER 50HZ
    xx1 = xx.flatten()
    cutoff = 40  # 40HZ
    xx1 = filter_data.butter_lowpass_filter(xx1, cutoff, 5000.0)

    #       SET WINDOW FOR ONSET DETECTING (SD)

    xx1 -= np.average(xx1)

    step = np.hstack((np.ones(len(xx1)), -1 * np.ones(len(xx1))))

    xx1_step = np.convolve(xx1, step, mode='valid')

    # get the peak of the convolution, its index

    step_index = np.argmin(xx1_step[400:800]) + 400  # yes, cleaner than np.where(xx1_step == xx1_step.max())[0][0]
    pick = np.argmax(xx1[0:step_index])

    window_size = 120
    onset_trs = 0.2

    x = round(pick / window_size)
    trs = int(x)
    onset1 = 0
    for i in range(1, trs):
        ss = np.std(xx1[(pick - window_size * i): (pick - window_size * (i - 1))])
        if ss < onset_trs:
            onset1 = pick - window_size * (i-1)
            break

    onset2 = np.argmin(xx1[150:pick]) + 150

    onset = max(onset1, onset2)
    
    # plt.axvline(x=pick, color='purple')
    plt.axvline(x=onset, color='green')
    onset_time = ((onset - (100 / sampling_rate)) / 5000)
    # pick_time = ((pick - (100 / sampling_rate)) / 5000)
    plt.text(onset, 2, str("%.3f" % round(onset_time, 3)), fontsize=6)

    #       SET WINDOW FOR PICK DETECTING (SD)
    # pick = np.argmax(xx1[(step_index+100):1000]) + step_index + 100

    step = np.argmax(xx1_step[500:900]) + 500
    pick = np.argmax(xx1[step:(step+100)]) + step
    plt.axvline(x=pick, color='orange')
    pick_time = ((pick - (100 / sampling_rate)) / 5000)
    plt.text(pick, 2, str("%.3f" % round(pick_time, 3)), fontsize=6)

    # plt.text((max_index+onset)/2, 0, str("%.3f" % round(dis,3)), fontsize=6)
    # max_index = max_index + 30

    #       CALCULATE DISTANCE
    distance = pick - onset
    distance_time = ((distance - (100 / sampling_rate)) / 5000)

    properties[counters-1][0] = sig_date
    properties[counters-1][1] = "%.3f" % round(onset_time, 3)
    properties[counters-1][2] = "%.3f" % round(pick_time, 3)
    properties[counters-1][3] = "%.3f" % round(distance_time, 3)

    return properties, xx1, cutoff, window_size, onset_trs


def write_excel(cluster_members, properties):
    wb = Workbook() 

    sheet1 = wb.add_sheet('Properties') 

    sheet1.write(0, 1, 'Date') 
    sheet1.write(0, 2, 'Onset(s)') 
    sheet1.write(0, 3, 'Pick(s)') 
    sheet1.write(0, 4, 'Distance(s)') 

    for i in range(cluster_members):
        for j in range(4):
            sheet1.write(i+1, j+1, properties[i][j])

    wb.save('../data/resultC0_1.xls')

