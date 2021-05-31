import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

COMPONENT_COUNT = 950

df = pd.read_csv('../3-vectorization/output/dataset_bigram.csv', sep=',', header=0)
X = df.drop('_class_', axis=1)

pca = PCA(n_components=COMPONENT_COUNT)
pca.fit(X)
print('{:<9}: {:.3f}%'.format('bigram', sum(pca.explained_variance_ratio_[:COMPONENT_COUNT]) * 100))

X_new = pca.transform(X)
X_new_cols = [ 'Comp{}'.format(index + 1) for index in range(X_new.shape[1]) ]
df_output = pd.DataFrame(data=X_new, columns=X_new_cols)
df_output['_class_'] = df['_class_']

df_output.to_csv('./output/4-{}.csv'.format('bigram'), sep=',', header=True, index=False)