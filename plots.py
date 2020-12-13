
import matplotlib.pyplot as plt
import numpy as np
from Analysis import analysisC1
from Analysis import analysisC0
import filter_data


def plotCluster(yi, y_pred, X_train, dates, sampling_rate, end, start, append_title='', analyse=False):
    cluster_number = 7
    plt.subplot(15, cluster_number, 1)
    plt.axis('off')
    counters = 0

    y_test = y_pred.tolist()
    cluster_members = y_test.count(yi)
    properties = [[0 for x in range(4)] for y in range(cluster_members)] 
    print(cluster_members)

    for xx in X_train[y_pred == yi]:
        counters = counters + 1
        if counters <= 13:
            plt.subplot(15, cluster_number, 1 + cluster_number*counters)
        if 13 < counters <= 26:
            plt.subplot(15, cluster_number, 2 + cluster_number*(counters - 13))
        if 26 < counters <= 39:
            plt.subplot(15, cluster_number, 3 + cluster_number*(counters - 26))
        if 39 < counters <= 52:
            plt.subplot(15, cluster_number, 4 + cluster_number*(counters - 39))
        if 52 < counters <= 65:
            plt.subplot(15, cluster_number, 5 + cluster_number*(counters - 52))
        if 65 < counters <= 78:
            plt.subplot(15, cluster_number, 6 + cluster_number*(counters - 65))
        if counters > 78:
            plt.subplot(15, cluster_number, 7 + cluster_number*(counters - 78)) 

        #       IDENTIFY WITH EACH SIGNAL DATE
        
        #       SHOW STIMULES ON PLOT
        plt.axvline(x=(100/sampling_rate), color='red')
        plt.axis('off')

        ind = np.where(X_train == xx)
        ind = ind[0][0]
        sig_date = dates[ind][0]
        plt.text(end-start, 1, str(sig_date), fontsize=8)

        max_first_tres = ''
        max_second_tres = ''
        onset_first_tres = ''
        onset_second_tres = ''
        max_window_size = ''
        second_title = ''
        xx1 = filter_data.butter_lowpass_filter(xx, 50, 5000)

        if analyse:
            if yi == 1:
                properties, xx1, onset_first_trs, onset_second_trs, fc, slicing_window_size, second_window_check_size \
                    = analysisC1.analyse_cluster1(xx, properties, sampling_rate, sig_date, counters)

                second_title = "\n onset_first_trs={}, onset_second_trs={} \n slicing_window={}, second_window={}" \
                               " \n cutoff frequency={}".format(onset_first_trs, onset_second_trs, slicing_window_size,
                                                                second_window_check_size, fc)

                analysisC1.writeExcel(cluster_members, properties)
            if yi == 0:
                properties, xx1, fc, window_size, onset_trs = analysisC0.analyse_cluster0(xx, properties, sampling_rate,
                                                                                          sig_date, counters)

                second_title = "\n window_size={}, onset_trs={} \n cutoff frequency={}".format(window_size, onset_trs,
                                                                                               fc)

                analysisC0.write_excel(cluster_members, properties)

        # analysisC1.writeExcel(cluster_members, properties)

        plt.plot(xx1)

    counters = 0
    plt.subplot(15, cluster_number, 1 + cluster_number*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')    

    plt.subplot(15, cluster_number, 2 + cluster_number*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')     
    
    plt.subplot(15, cluster_number, 3 + cluster_number*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')
    
    plt.subplot(15, cluster_number, 4 + cluster_number*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')     
    
    plt.subplot(15, cluster_number, 5 + cluster_number*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')

    plt.subplot(15, cluster_number, 6 + cluster_number*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')

    plt.subplot(15, cluster_number, 7 + cluster_number*14)
    plt.axis(xmin=-0.03, xmax=0.22)
    plt.axvline(x=0, color='red')

    plt.suptitle(append_title + "\n Sampling rate = "+ str(sampling_rate) + ", Period = " + str(start) + ":" +
                 str(end) + ", Cluster = " + str(yi) + "\n clusterMembers = " + str(cluster_members) + second_title)

    plt.show()

    if analyse:
        if yi == 1:
            analysisC1.analyse_distance(properties, cluster_members, feature=3)
            print(' ')
            analysisC1.dip_test(properties, cluster_members, feature=3)
        if yi == 0:
            # analysisC0.analyse_distance(properties, cluster_members, feature=3)
            print(' ')
            # analysisC0.dip_test(properties, cluster_members, feature=2)

    # plt.show()
    # if analyse:
    #     analysisC1.sub_cluster(yi, y_pred, properties, cluster_members, X_train, dates, sampling_rate, end, start)
