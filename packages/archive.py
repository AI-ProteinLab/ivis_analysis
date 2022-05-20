

'''
##### implied timescale comparison####
dataset = {
    "embeddings_ivis_unsupervised":embeddings_ivis_unsupervised,
    "embeddings_ivis_supervised":embeddings_ivis_supervised,
    "embeddings_pca":embeddings_pca,
    "embeddings_tsne":embeddings_tsne,
    "embeddings_tica":embeddings_tica
}

implied = []
for keys in dataset.keys():
    data = dataset[keys]
    tica_trajs = data.reshape(12, -1, 2)
    clusterer = MiniBatchKMeans(n_clusters=100, random_state=0, batch_size=50)
    clustered_trajs = clusterer.fit_transform(tica_trajs)
    msm = MarkovStateModel(lag_time=500, n_timescales=10)
    msm.fit(clustered_trajs)
    timescale = msm.timescales_[:10].reshape(1, -1)
    implied.append(timescale[0][0])

print([(name, value) for name, value in zip(list(dataset.keys()), implied)])
'''

'''
#### compare timescale #####
dataset = {
    "embeddings_ivis_unsupervised":embeddings_ivis_unsupervised,
    "embeddings_ivis_supervised":embeddings_ivis_supervised,
    "embeddings_pca":embeddings_pca,
    "embeddings_tsne":embeddings_tsne,
    "embeddings_tica":embeddings_tica
}

all_timescale = []
time_gap = np.array([i for i in range(0, 601, 100)])
for keys in dataset.keys():
    data = dataset[keys]
    tica_trajs = data.reshape(12, -1, 2)
    clusterer = MiniBatchKMeans(n_clusters=300, random_state=0, batch_size=100)
    clustered_trajs = clusterer.fit_transform(tica_trajs)
    timescale = np.array([0] * 15).reshape(1, -1)
    for time in time_gap:
        if time == 0:
            time = 1
        msm = MarkovStateModel(lag_time=time, n_timescales=15)
        msm.fit(clustered_trajs)
        implied = msm.timescales_[:15].reshape(1, -1)
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
plt.savefig("timescale.png")
'''



'''
#### prediction accuracy ######
for keys in dataset.keys():
    data = dataset[keys]
    tica_trajs = data.reshape(12, -1, 2)
    clusterer = MiniBatchKMeans(n_clusters=100, random_state=0, batch_size=50)
    clustered_trajs = clusterer.fit_transform(tica_trajs)
    msm = MarkovStateModel(lag_time=500, n_timescales=10)
    msm.fit(clustered_trajs)
    macro_state = 8
    pcca = PCCAPlus.from_msm(msm, n_macrostates=macro_state)
    macro_trajs = pcca.transform(clustered_trajs)
    y = np.array(macro_trajs)
    y = y.reshape(-1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    y_train = y_train.reshape(y_train.shape[0], )
    y_test = y_test.reshape(y_test.shape[0], )
    # model
    depth = 10
    clf = RandomForestClassifier(random_state=0, max_depth=depth, n_estimators=50)
'''

