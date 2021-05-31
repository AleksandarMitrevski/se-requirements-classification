import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

datasets = ['bow-12', 'bow-11', 'bow-2', 'tfidf-12', 'tfidf-11', 'tfidf-2']
component_counts = [300, 200, 250, 400, 300, 500] # based on slope of explained variance cumulative sum curves from 3-pca-investigate.py (98% - 99%)

print('** Explained Variance **')

def transform_dataset(name, component_count):
    df = pd.read_csv('./output/2-{}.csv'.format(name), sep=',', header=0)
    X = df.drop('_class_', axis=1)

    pca = PCA(n_components=component_count)
    pca.fit(X)
    print('{:<9}: {:.3f}%'.format(name, sum(pca.explained_variance_ratio_[:component_count]) * 100))

    X_new = pca.transform(X)
    X_new_cols = [ 'Comp{}'.format(index + 1) for index in range(X_new.shape[1]) ]
    df_output = pd.DataFrame(data=X_new, columns=X_new_cols)
    df_output['_class_'] = df['_class_']

    df_output.to_csv('./output/3-{}.csv'.format(name), sep=',', header=True, index=False)

for index, dataset in enumerate(datasets):
    transform_dataset(dataset, component_counts[index])