import math
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib

plt.figure()

plt.title('PCA Explained Variance Cumulative Sums')

df = pd.read_csv('../3-vectorization/output/dataset_bigram.csv', sep=',', header=0)
X = df.drop('_class_', axis=1)

components_count=min(X.shape[0], X.shape[1])
pca = PCA(n_components=components_count)
pca.fit(X)

explained_variance_cumsum = pd.Series(data=pca.explained_variance_ratio_).sort_values(ascending=False).cumsum()

plt.plot(explained_variance_cumsum)
plt.grid(color='#e0e0e0', which='major')
plt.grid(color='#f2f2f2', which='minor')

plt.show()