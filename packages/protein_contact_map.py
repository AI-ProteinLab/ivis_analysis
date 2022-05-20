import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
import math
import seaborn as sns


def contact_map(correlation_list):
    # fill in zeros
    full = np.array([1.0 for i in range(int(255 * 254 / 2))])
    count = 0
    for i in range(254):
        for j in range(i, 254):
            if i == j:
                full[count] = 0
            count += 1
    # fill in correlation_list values
    count = 0
    for i in range(full.shape[0]):
        if full[i] != 0:
            full[i] = correlation_list[count]
            count += 1
    # reshape into map
    c_map = np.zeros((254, 254))
    triu = np.triu_indices(254) # Find upper right indices of a triangular nxn matrix
    tril = np.tril_indices(254, -1) # Find lower left indices of a triangular nxn matrix
    c_map[triu] = full # Assign list values to upper right matrix
    c_map[tril] = c_map.T[tril] # Make the matrix symmetric
    return c_map


def plot_contact_map(values, name):
    plt.figure()
    ax = sns.heatmap(values, vmin=np.min(values), vmax=np.max(values), cmap='coolwarm', cbar=False)
    ax.set_yticks([])
    ax.set_xticks([])
    plt.savefig("./%s.png" %str(name))


def get_cor(original, reduced):
    scaler = StandardScaler(with_mean=False, with_std=False)
    original = scaler.fit_transform(original)
    reduced = scaler.fit_transform(reduced)
    cor = []
    for i in range(original.shape[1]):
        c, p = spearmanr(original[:, i], reduced[:, 0])
        if p <= 0.01:
            cor.append(c)
        else:
            cor.append(0)
    return cor


def get_residue(correlation_list):
    # only select important distance pairs
    mean = np.mean(correlation_list)
    std = np.std(correlation_list)
    count = 0
    for i in range(correlation_list.shape[0]):
        if mean - 3*std <= correlation_list[i] <= mean + 3*std:
            correlation_list[i] = 0
    # distribute distance into residues
    residue = [[0.0, i] for i in range(254)]
    count = 0
    for i in range(254):
        for j in range(i+1, 254):
            residue[i][0] += abs(correlation_list[count])
            residue[j][0] += abs(correlation_list[count])
            count += 1
    residue = sorted(residue, key = lambda x : -x[0])
    # convert i into residue number
    ans = [[], []]
    for i in range(254):
        if residue[i][1] <= 126:
            ans[0].append(residue[i][1] + 240)
        else:
            ans[1].append(residue[i][1] + 240 - 127)
    return ans


def residue_rank(data, top = 20):
    residue = [0 for i in range(254)]
    for i in range(254):
        for j in range(254):
            residue[i] += abs(data[i][j])
            residue[j] += abs(data[i][j])

    residue = np.array(residue)
    residue[:127] += residue[127:]
    residue[:127] /= 2
    resID = np.array([i for i in range(240, 367)]).reshape(-1, 1)
    res_rank = np.concatenate((residue[:127].reshape(-1,1), resID), axis = 1)
    res_rank = sorted(res_rank, key = lambda x : x[0], reverse = True)
    return res_rank

