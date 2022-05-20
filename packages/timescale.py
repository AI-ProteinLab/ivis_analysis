import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import seaborn as sns
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel


def plot():

    ###### load embeddings #####
    df_all = pd.read_csv("/users/haot/scratch/research/dimension/get_embedding_data/0/embedding_data.csv")
    embeddings_ivis_unsupervised = df_all[["embeddings_ivis_unsupervised_x", "embeddings_ivis_unsupervised_y"]].values
    embeddings_ivis_supervised = df_all[["embeddings_ivis_supervised_x", "embeddings_ivis_supervised_y"]].values
    embeddings_pca = df_all[["embeddings_pca_x", "embeddings_pca_y"]].values
    embeddings_tsne = df_all[["embeddings_tsne_x", "embeddings_tsne_y"]].values
    embeddings_tica = df_all[["embeddings_tica_x", "embeddings_tica_y"]].values


    #### compare timescale #####
    dataset = {
        "embeddings_ivis_unsupervised":embeddings_ivis_unsupervised,
        "embeddings_ivis_supervised":embeddings_ivis_supervised,
        "embeddings_pca":embeddings_pca,
        "embeddings_tsne":embeddings_tsne,
        "embeddings_tica":embeddings_tica
    }

    n_timescale = 5
    all_timescale = []
    time_gap = np.array([i for i in range(100, 701, 100)])
    for keys in dataset.keys():
        data = dataset[keys]
        tica_trajs = data.reshape(12, -1, 2)
        # 100, 42, 100 works except for tSNE
        # 50, 42, 50 works 
        if keys == "embeddings_tsne":
            n_cluster, batch_size = 50, 50
        else:
            n_cluster, batch_size = 100, 100
        clusterer = MiniBatchKMeans(n_clusters=n_cluster, random_state=42, batch_size=batch_size)
        clustered_trajs = clusterer.fit_transform(tica_trajs)
        timescale = np.array([0] * n_timescale).reshape(1, -1)
        for time in time_gap:
            if time == 0:
                time = 1
            msm = MarkovStateModel(lag_time=time, n_timescales=n_timescale)
            msm.fit(clustered_trajs)
            implied = msm.timescales_.reshape(1, -1)
            timescale = np.concatenate((timescale, implied), axis = 0)
        all_timescale.append(timescale[1:])


    # compare the first timescale of four models
    all_timescale = np.array(all_timescale)
    plt.figure()
    for i in range(5):
        plt.plot(time_gap/10, all_timescale[i, :, 0], ".--")

    plt.yscale("log")
    plt.xlabel("lag time (ns)")
    plt.ylabel("implied timescales (ns)")
    plt.legend(("ivis_unsupervised", "ivis_supervised", "pca", "tsne", "tica"))
    plt.savefig("/users/haot/scratch/research/dimension/result/timescale.png")

