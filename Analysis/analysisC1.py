import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy import signal
from scipy.stats import norm, uniform, gaussian_kde
import unidip.dip as dip

from xlwt import Workbook
import filter_data
import plots


def dip_test(properties, cluster_members, feature=3):
    data = []
    for ii in range(cluster_members):
        data.append(float(properties[ii][feature]))
    
    data = np.array(data)
    data = np.msort(data)
    intervals = dip.diptst(data)

    t_range = np.linspace(0, 0.15, 200)
    kde = gaussian_kde(data)
    plt.plot(t_range, kde(t_range)/100)
    plt.grid()
    plt.title('Pick PDF' + '\n p_value = ' + str(intervals[1]))

    plt.show()


def sub_cluster(cluster, y_pred, properties, cluster_members, X_train, dates, sampling_rate, end, start):
    array = []
    dates_array = []
    for xx in X_train[y_pred == cluster]:
        array.append(xx)
    for dd in dates[y_pred == cluster]:
        dates_array.append(dd)

    X_train = np.array(array)
    dates_array = np.array(dates_array)

    y_pred = []
    for i in range(cluster_members):
        if float(properties[i][3]) > float(0.030):
            y_pred.append(0)
        else:
            y_pred.append(1)
    y_pred = np.array(y_pred)

    plots.plotCluster(0, y_pred, X_train, dates_array, sampling_rate, end, start, appendTitle='')
    plots.plotCluster(1, y_pred, X_train, dates_array, sampling_rate, end, start, appendTitle='')


def analyse_distance(properties, cluster_members, feature=3):
    data = []
    for ii in range(cluster_members):
        data.append(float(properties[ii][feature]))

    num_bins = 8
    plt.ylabel('Cluster 1 signals')
    plt.xlabel('Cluster 1 onset')
    plt.title('Pick Histogram Plot \n Bins = ' + str(num_bins) + '\n Cluster 2')
    plt.grid()
    n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)

    plt.show()


def analyse_cluster1(xx, properties, sampling_rate, sigDate, counters):

    #       LOWPASS FILTER 50HZ
    xx1 = xx.flatten()
    cutoff = 40     # 40HZ
    xx1 = filter_data.butter_lowpass_filter(xx1, cutoff, 5000.0)

    #       SET WINDOW FOR ONSET DETECTING (SD)
    start_dev = int(100/sampling_rate)
    end_dev = int(100/sampling_rate + 20)
    onset_first_trs = 0.05
    onset_second_trs = 0.15
    slicing_window_size = 25
    second_window_check_size = 25

    #       ONSET DETECT ALGORITHM
    while 1:
        while 1:
            sam = xx1[start_dev:end_dev:1].flatten()
            if statistics.stdev(sam) > onset_first_trs:
                break
            start_dev = start_dev + slicing_window_size
            end_dev = end_dev + slicing_window_size

        sam = xx1[end_dev:(end_dev + second_window_check_size):1].flatten()
        if statistics.stdev(sam) > onset_second_trs:
            break
        else:
            start_dev = start_dev + slicing_window_size
            end_dev = end_dev + slicing_window_size

    # onset = (start_dev+end_dev)/2
    onset = start_dev

    plt.axvline(x=onset, color='green')
    onset_time = ((onset-(100/sampling_rate))/5000)
    plt.text(onset, 2, str("%.3f" % round(onset_time, 3)), fontsize=6)

    #       SET WINDOW FOR PICK DETECTING (SD)
    start_dev = start_dev + 150
    end_dev = end_dev + 150
    max_first_trs = 0.1
    max_second_trs = 0.1
    slicing_window_size = 70
    second_window_check_size = 70
    max_window_size = 100
    while 1:
        while 1:
            sam = xx1[start_dev:end_dev].flatten()
            if statistics.stdev(sam) < max_first_trs:
                break
            start_dev = start_dev + slicing_window_size
            end_dev = end_dev + slicing_window_size
            if start_dev > 100:
                break

        if start_dev > 100:
            break
        sam = xx1[end_dev:end_dev + second_window_check_size].flatten()
        if statistics.stdev(sam) < max_second_trs:
            break
        else:
            start_dev = start_dev + slicing_window_size
            end_dev = end_dev + slicing_window_size

    xx_onset = xx1[start_dev:start_dev + max_window_size]
    pick1 = np.where(xx_onset == np.amax(xx_onset))[0][0] + start_dev

    if sigDate == u'130103':
        pick1 = 550

    #   SECOND APPROACH

    # xx1 -= np.average(xx1)
    #
    # step = np.hstack((np.ones(len(xx1)), -1 * np.ones(len(xx1))))
    #
    # xx1_step = np.convolve(xx1, step, mode='valid')
    #
    # # get the peak of the convolution, its index
    #
    # step_index = np.argmax(xx1_step)
    # pick2 = np.argmax(xx1[step_index-100:step_index+100]) + step_index - 100

    # pick = np.mean([pick1, pick2])
    pick = pick1
    plt.axvline(x=pick, color='orange')
    pick_time = ((pick - (100 / sampling_rate)) / 5000)

    plt.text(onset, 2, str("%.3f" % round(pick_time, 3)), fontsize=6)

    # #       CAL DISTANCE
    distance = pick_time - onset_time
    plt.text((pick_time + onset)/2, 0, str("%.3f" % round(distance, 3)), fontsize=6)

    #       FILL PROPERTIES OF A SIGNAL
    properties[counters-1][0] = sigDate
    properties[counters-1][1] = "%.3f" % round(onset_time, 3)
    properties[counters-1][2] = "%.3f" % round(pick_time, 3)
    properties[counters-1][3] = "%.3f" % round(distance, 3)

    return properties, xx1, onset_first_trs, onset_second_trs, cutoff, slicing_window_size, second_window_check_size


def writeExcel(cluster_members, properties):
    wb = Workbook() 

    sheet1 = wb.add_sheet('Properties') 

    sheet1.write(0, 1, 'Date') 
    sheet1.write(0, 2, 'Onset(s)') 
    sheet1.write(0, 3, 'Pick(s)') 
    sheet1.write(0, 4, 'Distance(s)') 

    for i in range(cluster_members):
        for j in range(4):
            sheet1.write(i+1, j+1, properties[i][j])

    wb.save('../data/result.xls')

