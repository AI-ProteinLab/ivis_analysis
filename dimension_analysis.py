import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from matplotlib.pyplot import text
import pandas as pd
import seaborn as sns
import math
from ivis import Ivis
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr, pearsonr
from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from msmbuilder.decomposition import tICA
from msmbuilder.cluster import MiniBatchKMeans
from msmbuilder.msm import MarkovStateModel
from msmbuilder.lumping import PCCAPlus
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.ensemble import RandomForestClassifier

# self designed packages
import sys
sys.path.insert(0, "/users/haot/scratch/research/dimension/packages")
import traj_data
import Mantel_test as mantel
import protein_contact_map as pcm
import preserve_pcc as preserve
import cluster_rmsd
import timescale
import information

start = 0

###################### get data ######################
X, y, gap, cutoff, num_of_sample = traj_data.get_traj_data(gap=5, cutoff=1.0, start = start)


##### ivis #####
# unsupervised 
metrics_ivis_2d_unsupervised = {
    "n_epochs_without_progress" : 10, 
    "supervision_weight": 0.0, 
    "embedding_dims": 2, 
    "distance": "euclidean", 
    "margin": 1,
    "model":"maaten",
    "supervision_metric": "sparse_categorical_crossentropy",
    "k" : 100
}

ivis_unsupervised = Ivis(**metrics_ivis_2d_unsupervised)
ivis_unsupervised.fit(X, y)
embeddings_ivis_unsupervised = ivis_unsupervised.transform(X)


# supervised
metrics_ivis_2d_supervised = {
    "n_epochs_without_progress" : 10, 
    "supervision_weight": 0.1, 
    "embedding_dims": 2, 
    "distance": "euclidean", 
    "margin": 1,
    "model":"maaten",
    "supervision_metric": "sparse_categorical_crossentropy",
    "k" : 100
}

ivis_supervised = Ivis(**metrics_ivis_2d_supervised)
ivis_supervised.fit(X, y)
embeddings_ivis_supervised = ivis_supervised.transform(X)

##### PCA #####
pca = PCA(n_components = 2, random_state=0)
pca.fit(X)
embeddings_pca = pca.transform(X)

##### t-SNE #####
model_tsne = TSNE(learning_rate = 200, n_components = 2, perplexity = 5, n_iter = 300, verbose = 0, n_jobs=16)
embeddings_tsne = model_tsne.fit_transform(X)

##### tICA #####
tica_data = X.reshape(12, int(X.shape[0]/12), X.shape[1])
tica_model = tICA(lag_time=1, n_components=2, kinetic_mapping=True)
tica_model.fit(tica_data)
tica_trajs = tica_model.transform(tica_data)
embeddings_tica = np.concatenate(tica_trajs)


###################### load embeddings ######################
df_all = pd.read_csv("./get_embedding_data/%s/embedding_data.csv" %start)
embeddings_ivis_unsupervised = df_all[["embeddings_ivis_unsupervised_x", "embeddings_ivis_unsupervised_y"]].values
embeddings_ivis_supervised = df_all[["embeddings_ivis_supervised_x", "embeddings_ivis_supervised_y"]].values
embeddings_pca = df_all[["embeddings_pca_x", "embeddings_pca_y"]].values
embeddings_tsne = df_all[["embeddings_tsne_x", "embeddings_tsne_y"]].values
embeddings_tica = df_all[["embeddings_tica_x", "embeddings_tica_y"]].values


###################### analysis ######################

###################### check model RMSD ######################
rmsd_ivis_un = cluster_rmsd.cluster_rmsd(5, embeddings_ivis_unsupervised)
rmsd_ivis_sp = cluster_rmsd.cluster_rmsd(5, embeddings_ivis_supervised)
rmsd_pca = cluster_rmsd.cluster_rmsd(5, embeddings_pca)
rmsd_tsne = cluster_rmsd.cluster_rmsd(5, embeddings_tsne)
rmsd_tica = cluster_rmsd.cluster_rmsd(5, embeddings_tica)


###################### spearson ###################### -> results on spearson folder
dist_orig = np.square(euclidean_distances(X, X)).flatten()
dist_ivis_unsupervised = np.square(euclidean_distances(embeddings_ivis_unsupervised, embeddings_ivis_unsupervised)).flatten()
dist_ivis_supervised = np.square(euclidean_distances(embeddings_ivis_supervised, embeddings_ivis_supervised)).flatten()
dist_pca = np.square(euclidean_distances(embeddings_pca, embeddings_pca)).flatten()
dist_tsne = np.square(euclidean_distances(embeddings_tsne, embeddings_tsne)).flatten()
dist_tica = np.square(euclidean_distances(embeddings_tica, embeddings_tica)).flatten()

coef_ivis_unsupervised, p_ivis = spearmanr(dist_orig, dist_ivis_unsupervised)
coef_ivis_supervised, p_ivis = spearmanr(dist_orig, dist_ivis_supervised)
coef_pca, p_pca = spearmanr(dist_orig, dist_pca)
coef_tsne, p_tsne = spearmanr(dist_orig, dist_tsne)
coef_tica, p_tica = spearmanr(dist_orig, dist_tica)

print("spearson coefficient")
print("ivis_unsupervised : %f, ivis_supervised : %f, pca : %f, tsne : %f, tica : %f" %(coef_ivis_unsupervised, coef_ivis_supervised, coef_pca, coef_tsne, coef_tica))


###################### Mantel test ######################

original_distances = mantel.distance_matrix(X, num_of_sample)
ivis_unsupervised_distances = mantel.distance_matrix(embeddings_ivis_unsupervised, num_of_sample)
ivis_supervised_distances = mantel.distance_matrix(embeddings_ivis_supervised, num_of_sample)
pca_distances = mantel.distance_matrix(embeddings_pca, num_of_sample)
tsne_distances = mantel.distance_matrix(embeddings_tsne, num_of_sample)
tica_distances = mantel.distance_matrix(embeddings_tica, num_of_sample)

mantel_ivis_unsupervised = mantel.test(original_distances, ivis_unsupervised_distances, perms=10000, method='pearson')
mantel_ivis_supervised = mantel.test(original_distances, ivis_supervised_distances, perms=10000, method='pearson')
mantel_pca = mantel.test(original_distances, pca_distances, perms=10000, method='pearson')
mantel_tsne = mantel.test(original_distances, tsne_distances, perms=10000, method='pearson')
mantel_tica = mantel.test(original_distances, tica_distances, perms=10000, method='pearson')

print("Mantel test result")
print("ivis_unsupervised : %f, ivis_supervised : %f, pca : %f, tsne : %f, tica : %f" %(mantel_ivis_unsupervised[0], mantel_ivis_supervised[0], mantel_pca[0], mantel_tsne[0], mantel_tica[0]))

'''
#### mahalanobis distances #####
# !!! sample size must be greater than feature size !!!
from sklearn.metrics.pairwise import pairwise_distances

orig_dist = np.square(pairwise_distances(X, X, metric = 'mahalanobis')).flatten()
ivis_supervised_dist = np.square(pairwise_distances(embeddings_ivis_supervised, embeddings_ivis_supervised, metric = 'mahalanobis')).flatten()
ivis_unsupervised_dist = np.square(pairwise_distances(embeddings_ivis_unsupervised, embeddings_ivis_unsupervised, metric = 'mahalanobis')).flatten()
pca_dist = np.square(pairwise_distances(embeddings_pca[:num_of_sample*3], embeddings_pca[num_of_sample*3:], metric = 'mahalanobis')).flatten()
tsne_dist = np.square(pairwise_distances(embeddings_tsne[:num_of_sample*3], embeddings_tsne[num_of_sample*3:], metric = 'mahalanobis')).flatten()

coef_ivis_unsupervised, p_ivis_unsupervised = spearmanr(orig_dist, ivis_unsupervised_dist)
coef_ivis_supervised, p_ivis_supervised = spearmanr(orig_dist, ivis_supervised_dist)
coef_pca, p_pca = spearmanr(orig_dist, pca_dist)
coef_tsne, p_tsne = spearmanr(orig_dist, tsne_dist)
'''

###################### plot reduced embedding data map ######################
def plot_reduced_2d_map(data, name, start):
    plt.figure()
    for i in range(4):
        plt.scatter(data[num_of_sample*i*3:num_of_sample*(i+1)*3, 0], data[num_of_sample*i*3:num_of_sample*(i+1)*3, 1], s=0.4)
    plt.legend(("dark", "traisition dark", "transition light", "light"))
    plt.savefig("/users/haot/scratch/research/dimension/result/%s_%s.png" %(name, start))


plot_reduced_2d_map(embeddings_ivis_unsupervised, "embeddings_ivis_unsupervised", start)
plot_reduced_2d_map(embeddings_ivis_supervised, "embeddings_ivis_supervised", start)
plot_reduced_2d_map(embeddings_pca, "embeddings_pca", start)
plot_reduced_2d_map(embeddings_tica, "embeddings_tica", start)
plot_reduced_2d_map(embeddings_tsne, "embeddings_tsne", start)


###################### MSM implied timescale ######################
import sys
sys.path.insert(0, "/users/haot/scratch/research/dimension/packages")
import timescale

timescale.plot()


###################### protein contact map ######################
col = []
for start in range(5):
    ivis_unsupervised = Ivis().load_model("get_embedding_data/%s/ivis_supervised" %start)
    contact_ivis_unsupervised = ivis_unsupervised.model_.get_weights()[0][:, -1]
    map_ivis_unsupervised = pcm.contact_map(contact_ivis_unsupervised)
    res_imp = pcm.residue_rank(data = map_ivis_unsupervised, top = 20)
    res_imp = np.array(res_imp)
    res_imp[:, 0] = res_imp[:, 0] / sum(res_imp[:, 0]) * 100
    col.append(np.array(sorted(res_imp, key = lambda x : x[1])))

col[0][:, 0] = col[0][:, 0] + col[1][:, 0] + col[2][:, 0] + col[3][:, 0] + col[4][:, 0]
res_imp = col[0]
res_imp[:, 0] = res_imp[:, 0] / sum(res_imp[:, 0]) * 100
res_imp = sorted(res_imp, key = lambda x : x[0], reverse = True)

ivis_supervised = Ivis().load_model("get_embedding_data/%s/ivis_supervised" %start)
contact_ivis_supervised = ivis_supervised.model_.get_weights()[0][:, -1]
map_ivis_supervised = pcm.contact_map(contact_ivis_supervised)
pcm.plot_contact_map(map_ivis_supervised, "contact_map_ivis_supervised_%s" %start)

contact_pca = pca.components_[0]
map_pca = pcm.contact_map(contact_pca)
pcm.plot_contact_map(map_pca, "contact_map_pca_%s" %start)

tica_component = pickle.load(open("get_tica_component/%s/tica_component.pkl" %start, "rb"))
contact_tica = tica_component[1]
map_tica = pcm.contact_map(contact_tica)
pcm.plot_contact_map(map_tica, "contact_map_tica_%s" %start)

embeddings_tsne = pickle.load(open("embeddings_tsne_1d.pkl", "rb"))
contact_tsne = pcm.get_cor(X, embeddings_tsne)
map_tsne = pcm.contact_map(contact_tsne)
pcm.plot_contact_map(map_tsne, "contact_map_tsne_%s" %start)


## get important residue
ivis_unsupervised_residue = pcm.get_residue(contact_ivis_unsupervised)
ivis_supervised_residue = pcm.get_residue(contact_ivis_supervised)
pca_residue = pcm.get_residue(contact_pca)
tica_residue = pcm.get_residue(contact_tica)
tsne_residue = pcm.get_residue(contact_tsne)



###################### preserve distance plot ######################
np.random.seed(0)
random_index = np.random.randint(low = 0, high = num_of_sample * 12 + 1, size = 1000)
boxplot_bin = 10

# calculate preserve distance matrix
preserve_ori = preserve.distance_matrix(random_index, X)
preserve_ivis_s = preserve.distance_matrix(random_index, embeddings_ivis_supervised)
preserve_ivis_u = preserve.distance_matrix(random_index, embeddings_ivis_unsupervised)
preserve_pca = preserve.distance_matrix(random_index, embeddings_pca)
preserve_tica = preserve.distance_matrix(random_index, embeddings_tica)
preserve_tsne = preserve.distance_matrix(random_index, embeddings_tsne)

# plot
preserve.plot(random_index, preserve_ori, boxplot_bin, "embeddings_original_%s" %start, 1, 1)
preserve.plot(random_index, preserve_ivis_s, boxplot_bin, "embeddings_ivis_supervised_%s" %start, 
    pearsonr(preserve_ori[0], preserve_ivis_s[0])[0], pearsonr(preserve_ori[1], preserve_ivis_s[1])[0])
preserve.plot(random_index, preserve_ivis_u, boxplot_bin, "embeddings_ivis_unsupervised_%s" %start, 
    pearsonr(preserve_ori[0], preserve_ivis_u[0])[0], pearsonr(preserve_ori[1], preserve_ivis_u[1])[0])
preserve.plot(random_index, preserve_pca, boxplot_bin, "embeddings_pca_%s" %start, 
    pearsonr(preserve_ori[0], preserve_pca[0])[0], pearsonr(preserve_ori[1], preserve_pca[1])[0])
preserve.plot(random_index, preserve_tica, boxplot_bin, "embeddings_tica_%s" %start, 
    pearsonr(preserve_ori[0], preserve_tica[0])[0], pearsonr(preserve_ori[1], preserve_tica[1])[0])
preserve.plot(random_index, preserve_tsne, boxplot_bin, "embeddings_tsne_%s" %start, 
    pearsonr(preserve_ori[0], preserve_tsne[0])[0], pearsonr(preserve_ori[1], preserve_tsne[1])[0])



###################### prediction accuracy from embedding data ######################
## see result on model_prediction folder


###################### Information Content ######################
info_ivis_un_ave, info_ivis_su_ave, info_pca_ave, info_tsne_ave, info_tica_ave = [], [], [], [], []

for start in range(5):
    df_all = pd.read_csv("./get_embedding_data/1d/%s/embedding_data.csv" %start)
    embeddings_ivis_unsupervised = df_all[["embeddings_ivis_unsupervised"]].values
    embeddings_ivis_supervised = df_all[["embeddings_ivis_supervised"]].values
    embeddings_pca = df_all[["embeddings_pca"]].values
    embeddings_tsne = df_all[["embeddings_tsne"]].values
    embeddings_tica = df_all[["embeddings_tica"]].values
    info_ivis_un_ave.append(information.get_probability(embeddings_ivis_unsupervised, 100)[1])
    info_ivis_su_ave.append(information.get_probability(embeddings_ivis_supervised, 100)[1])
    info_pca_ave.append(information.get_probability(embeddings_pca, 100)[1])
    info_tsne_ave.append(information.get_probability(embeddings_tsne, 100)[1])
    info_tica_ave.append(information.get_probability(embeddings_tica, 100)[1])


print(info_ivis_un_ave, info_ivis_su_ave, info_pca_ave, info_tsne_ave, info_tica_ave)
