import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns

sns.set_style('ticks')



features = pd.read_csv("face.csv")

for i in range(6):
    features['filename'][i] = 'cozinhando'

for i in range(6, 12):
    features['filename'][i] = 'emo'

for i in range(12, 18):
    features['filename'][i] = 'jogador'

for i in range(18,24):
    features['filename'][i] = 'ovo'

for i in range(24, 30):
    features['filename'][i] = 'violao'


#data_subset = features.loc[:, 'f0':'f127']
data_subset = features.loc[:, 'f0':'f127']

filenames = features.loc[:, 'filename']

# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(data_subset)
#
# data_subset['pca-one'] = pca_result[:,0]
# data_subset['pca-two'] = pca_result[:,1]

# data_subset = cosine_similarity(data_subset, data_subset)
# data_subset = pd.DataFrame(data_subset)
# data_subset = data_subset + 1
# print(data_subset)

time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)

data_subset.insert(0, 'filename', filenames)

data_subset['tsne-2d-one'] = tsne_results[:,0]
data_subset['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

ax1 = plt.subplot(1, 1, 1)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="filename",
    data=data_subset,
    legend="full",
    alpha=1,
    ax=ax1
)


# sns.scatterplot(
#     x="pca-one", y="pca-two",
#     hue='filename',
#     data=data_subset,
#     legend='full',
#     alpha=1,
#     ax=ax1
# )

# lmplot = sns.lmplot(
# x="pca-one",
# y="pca-two",
# hue='filename',
# data=data_subset
# )
#
# lmplot.savefig('lmplot.png')
plt.savefig('scatterTSNE.png')
