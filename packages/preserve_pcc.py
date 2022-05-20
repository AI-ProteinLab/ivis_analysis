import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns


def distance_matrix(random_index, data):
    l1_distance = []
    l2_distance = []
    for i in range(random_index.shape[0]):
        for j in range(i+1, random_index.shape[0]):
            l1 = np.sum(np.absolute(data[random_index[i]] - data[random_index[j]]))
            l2 = np.sum(np.square(data[random_index[i]] - data[random_index[j]]))
            l1_distance.append(l1)
            l2_distance.append(l2)
    return np.array(l1_distance), np.array(l2_distance)


def plot(random_index, data, bin_length, name, pearson_l1 = 0.0, pearson_l2 = 0.0):
    l1, l2 = data
    l1, l2  = np.sort(l1).reshape(bin_length, -1), np.sort(l2).reshape(bin_length, -1)
    l1_ave, l2_ave = np.average(l1, axis = 1), np.average(l2, axis = 1)
    l1, l2 = pd.DataFrame(l1.T), pd.DataFrame(l2.T)
    plt.figure(figsize=(9,5))
    plt.subplot(1,2,1)
    ax1 = sns.boxplot(data=l1)
    plt.plot(l1_ave, "r-")
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.ylabel("L1 distance")
    text(0.1, 0.9, s = "r = " + str(round(pearson_l1, 2)), transform=ax1.transAxes)
    plt.subplot(1,2,2)
    ax2 = sns.boxplot(data=l2)
    plt.plot(l2_ave, "r-")
    ax2.set_xticks([])
    ax2.set_yticks([])
    plt.ylabel("L2 distance")
    text(0.1, 0.9, s = "r = " + str(round(pearson_l2, 2)), transform=ax2.transAxes)
    plt.savefig("/users/haot/scratch/research/dimension/result/boxplot_%s.png" %name)
