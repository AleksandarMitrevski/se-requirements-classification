from math import log
import numpy as np
import pandas as pd

df = pd.read_csv('../2-preprocessing/output/dataset_normalized.csv', sep=',', header=0, quotechar = '"', doublequote=True)

# manual vectorization

features_set = set()
vector_list = [None] * df.shape[0]

for index, row in df.iterrows():
    text = row['RequirementText']
    tokens = text.split(' ')
    row_vector = {}
    tokens_count = len(tokens)
    for i in range(tokens_count):
        first_token = tokens[i]
        if (i + 1 < tokens_count):
            second_token = tokens[i + 1]
            bigram = first_token + ' ' + second_token
            features_set.add(bigram)
            if bigram in row_vector:
                row_vector[bigram] = row_vector[bigram] + 1
            else:
                row_vector[bigram] = 1
    vector_list[index] = row_vector

features_sorted = list(features_set)
features_sorted.sort()
features_sorted_lookup_map = { value: index for index, value in enumerate(features_sorted) }

vector_matrix_bigram = np.zeros((df.shape[0], len(features_set)), dtype=np.int16)

for i in range(len(vector_list)):
    row_dict = vector_list[i]
    for k, v in row_dict.items():
        vector_matrix_bigram[i, features_sorted_lookup_map[k]] = v

bigram_df = pd.DataFrame(data=vector_matrix_bigram, columns=features_sorted)
bigram_df['_class_'] = df['_class_']
bigram_df.to_csv('./output/dataset_bigram.csv', sep=',', header=True, index=False)
