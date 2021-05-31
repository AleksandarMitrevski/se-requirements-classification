from math import log
import numpy as np
import pandas as pd

df = pd.read_csv('../2-preprocessing/output/dataset_normalized.csv', sep=',', header=0, quotechar = '"', doublequote=True)

# manual vectorization; not using CountVectorizer and TfidfVectorizer from sklearn.feature_extraction

features_set = set()
vector_list = [None] * df.shape[0]

for index, row in df.iterrows():
    text = row['RequirementText']
    tokens = text.split(' ')
    row_vector = {}
    for token in tokens:
        features_set.add(token)
        if token in row_vector:
            row_vector[token] = row_vector[token] + 1
        else:
            row_vector[token] = 1
    vector_list[index] = row_vector

features_sorted = list(features_set)
features_sorted.sort()
features_sorted_lookup_map = { value: index for index, value in enumerate(features_sorted) }

vector_matrix_bow = np.zeros((df.shape[0], len(features_set)), dtype=np.int16)

for i in range(len(vector_list)):
    row_dict = vector_list[i]
    for k, v in row_dict.items():
        vector_matrix_bow[i, features_sorted_lookup_map[k]] = v

bow_df = pd.DataFrame(data=vector_matrix_bow, columns=features_sorted)
bow_df['_class_'] = df['_class_']
bow_df.to_csv('./output/dataset_bow.csv', sep=',', header=True, index=False)

vector_matrix_tfidf = np.zeros((df.shape[0], len(features_set)))
row_sums_bow = vector_matrix_bow.sum(axis=1)
col_sums_bow = vector_matrix_bow.sum(axis=0)

for i in range(vector_matrix_bow.shape[0]):
    for j in range(vector_matrix_bow.shape[1]):
        if vector_matrix_bow[i, j] == 0:    # minor optimization
            continue
        tf = vector_matrix_bow[i, j] / row_sums_bow[i]
        idf = log(vector_matrix_bow.shape[0] / col_sums_bow[j])
        vector_matrix_tfidf[i, j] = tf * idf

tfidf_df = pd.DataFrame(data=vector_matrix_tfidf, columns=features_sorted)
tfidf_df['_class_'] = df['_class_']
tfidf_df.to_csv('./output/dataset_tfidf.csv', sep=',', header=True, index=False)

# bow_df and tfidf_df are (very) sparse matrices of shape (969, 1524)
