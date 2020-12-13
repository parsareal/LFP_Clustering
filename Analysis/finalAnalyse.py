import csv
pros = []
with open("../data/result.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    counter = 0
    for lines in csv_reader:
        if counter != 0:
            pros.append(lines[3])
        counter = 1

with open("../data/resultC0_1.csv", "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    counter = 0
    outliers = [u'121010', u'091110', u'091109', u'091106']
    for lines in csv_reader:
        if counter != 0:
            if not lines[1] in outliers:
                pros.append(lines[3])
        counter = 1

from scipy.stats import norm, uniform, gaussian_kde
import numpy as np
from unidip import dip
import matplotlib.pyplot as plt

data = []

for ii in range(len(pros)):
    data.append(float(pros[ii]))

data = np.array(data)
data = np.msort(data)
# y = np.arange(1, len(data) + 1) / len(data)

# intervals = dip.diptst(data)
# t_range = np.linspace(0, 0.15, 200)
# kde = gaussian_kde(data)
# plt.plot(t_range, kde(t_range)/100)
# plt.title('All Data Onset PDF' + '\n p_value = ' + str(intervals[1]))

num_bins = 15
n, bins, patches = plt.hist(data, num_bins, facecolor='blue', alpha=0.5)
plt.title('Pick Histogram Plot \n Bins = ' + str(num_bins) + '\n Cluster 0,1')

plt.grid()
plt.show()
