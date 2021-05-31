import pandas as pd

import os, sys
sys.path.insert(0, os.path.abspath('..'))
from utils import utils

bow_df = pd.read_csv('../3-vectorization/output/dataset_bow.csv', sep=',', header=0)
tfidf_df = pd.read_csv('../3-vectorization/output/dataset_tfidf.csv', sep=',', header=0)

# 12-class datasets (no changes, rename for consistency)
bow_twelve_df = bow_df
bow_twelve_df.to_csv('./output/1-bow-12.csv', sep=',', header=True, index=False)

tfidf_twelve_df = tfidf_df
tfidf_twelve_df.to_csv('./output/1-tfidf-12.csv', sep=',', header=True, index=False)

# 11-class datasets
bow_eleven_df = bow_df.copy()
bow_eleven_df = bow_eleven_df[bow_eleven_df['_class_'] != 'F']
bow_eleven_df.to_csv('./output/1-bow-11.csv', sep=',', header=True, index=False)

tfidf_eleven_df = tfidf_df.copy()
tfidf_eleven_df = tfidf_eleven_df[tfidf_eleven_df['_class_'] != 'F']
tfidf_eleven_df.to_csv('./output/1-tfidf-11.csv', sep=',', header=True, index=False)

# 2-class datasets
binarize_class = lambda entry: 'F' if entry == 'F' else 'NF'

bow_two_df = bow_df.copy()
bow_two_df['_class_'] = bow_two_df['_class_'].apply(utils.binarize_class_variable)
bow_two_df.to_csv('./output/1-bow-2.csv', sep=',', header=True, index=False)

tfidf_two_df = tfidf_df.copy()
tfidf_two_df['_class_'] = tfidf_two_df['_class_'].apply(utils.binarize_class_variable)
tfidf_two_df.to_csv('./output/1-tfidf-2.csv', sep=',', header=True, index=False)
