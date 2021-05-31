import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from gensim.models import KeyedVectors

WORD2VEC_EMBEDDINGS_VECTOR_SIZE = 300

def load_dataset(dataset_dir, dataset, pca_version=True):
    version_prefix = 3 if pca_version else 2
    df = pd.read_csv('{}/{}-{}.csv'.format(dataset_dir, version_prefix, dataset), sep=',', header=0)
    return df

def cv_split(df, K):
    ###Returns training and test DataFrame splits for k-fold cross validation###

    # copy the dataframe with randomized rows
    clone_df = df.sample(frac=1).reset_index(drop=True)

    # split and return
    return np.array_split(clone_df, K)

def load_dataset_word2vec(word2vec_model, dataset_dir, mode, verbose=True):
    df = pd.read_csv('{}/dataset_normalized.csv'.format(dataset_dir), sep=',', header=0)
    if mode == '11-class':
        df = df[df['_class_'] != 'F'].reset_index(drop=True)
    
    num_rows = df.shape[0]
    DFRequirementText = df['RequirementText']
    DFClass = df['_class_'] if not mode == '2-class' else df['_class_'].apply(binarize_class_variable)

    if verbose:
        print('utils | Loading word2vec model...')
    w2v_model = KeyedVectors.load_word2vec_format(word2vec_model, binary=True)
    if verbose:
        print('utils | Word2vec model loaded.')

    # build token list and find maximum sequence length
    tokens_list = [''] # the empty element is to account for "mask_zero=True" in Sequential
    max_sequence_length = 0
    for i in range(num_rows):
        tokens = DFRequirementText[i].split(' ')
        num_tokens = len(tokens)

        if max_sequence_length < num_tokens:
            max_sequence_length = num_tokens

        for token in tokens:
            if token not in tokens_list: 
                tokens_list.append(token)

    # build dataset
    dataset = pd.DataFrame(data=np.zeros((num_rows, max_sequence_length)))
    dataset['_class_'] = DFClass
    tokens_list_map = { value: index for index, value in enumerate(tokens_list) }
    for i in range(num_rows):
        tokens = DFRequirementText[i].split(' ')
        num_tokens = len(tokens)

        for j, token in enumerate(tokens):
            dataset.iloc[i, j] = tokens_list_map[token]

    # build embeddings matrix
    tokens_count = len(tokens_list)
    embeddings_matrix = np.zeros((tokens_count, WORD2VEC_EMBEDDINGS_VECTOR_SIZE))
    for i in range(1, tokens_count):
        token = tokens_list[i]
        if token in w2v_model.vocab:
            embeddings_matrix[i, :] = w2v_model[token]
        # else it is a zero vector; there is ongoing research about finding other approaches

    return (embeddings_matrix, max_sequence_length, dataset)

def estimate_model_performance(model, X_test, Y_test):
    predictions = model.predict(X_test)

    # computing the confusion matrix is feasible (for both the binary and multiclass cases), but the result is not used in this project
    metrics_prf = precision_recall_fscore_support(Y_test, predictions, average='weighted')
    accuracy = accuracy_score(Y_test, predictions)

    return (metrics_prf[0], metrics_prf[1], metrics_prf[2], accuracy)

def binarize_class_variable(entry):
    return 'F' if entry == 'F' else 'NF'
