import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import os, sys
sys.path.insert(0, os.path.abspath("../.."))
from utils import utils

K = 10  # k-fold cross validation

class_names = ['F', 'NF', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']

def transform_class_names_to_labels(arr):
    f = np.vectorize(lambda x: class_names.index(x))
    return f(arr)

def transform_labels_to_class_names(arr):
    f = np.vectorize(lambda x: class_names[x])
    return f(arr)

def evaluate_two_step_model(dataset, output_filename, selected_k_binary, selected_k_nfr):
    output_info = '** {} **'.format(dataset)
    output = open('../results/{}.txt'.format(output_filename), "w")

    training_results = [0, 0, 0, 0]
    test_results = [0, 0, 0, 0]

    df = utils.load_dataset('../../4-feature-selection/output', dataset, True)
    random_split = utils.cv_split(df, K)

    for j in range(K):
        test = random_split[j]
        training_list = random_split[0:j] + random_split[j+1:K]
        training = pd.concat(training_list)

        binary_training = training.copy(deep=True)
        binary_training['_class_'] = binary_training['_class_'].apply(utils.binarize_class_variable)
        binary_test = test.copy(deep=True)
        binary_test['_class_'] = binary_test['_class_'].apply(utils.binarize_class_variable)

        nfr_training = training.copy(deep=True)
        nfr_training = nfr_training[nfr_training['_class_'] != 'F']

        X_train_binary = binary_training.drop('_class_', axis=1)
        Y_train_binary = transform_class_names_to_labels(binary_training['_class_'])
        X_test_binary = binary_test.drop('_class_', axis=1)
        #Y_test_binary = transform_class_names_to_labels(binary_test['_class_'])
        model_binary = KNeighborsClassifier(n_neighbors=selected_k_binary, weights='distance')
        model_binary.fit(X_train_binary, Y_train_binary)
        
        X_train_nfr = nfr_training.drop('_class_', axis=1)
        Y_train_nfr = transform_class_names_to_labels(nfr_training['_class_'])
        model_nfr = KNeighborsClassifier(n_neighbors=selected_k_nfr, weights='distance')
        model_nfr.fit(X_train_nfr, Y_train_nfr)

        # evaluation on training set
        current_predictions = model_binary.predict(X_train_binary)
        current_predictions = transform_labels_to_class_names(current_predictions)
        nfr_predictions_mask = current_predictions == 'NF'
        nfr_predictions_mask_indeces = np.where(current_predictions == 'NF')[0]

        two_step_samples = training[nfr_predictions_mask]
        X_two_step = two_step_samples.drop('_class_', axis=1)
        #Y_two_step = transform_class_names_to_labels(two_step_samples['_class_'])

        nfr_predictions = model_nfr.predict(X_two_step)
        nfr_predictions = transform_labels_to_class_names(nfr_predictions)
        for i in range(len(nfr_predictions_mask_indeces)):
            index = nfr_predictions_mask_indeces[i]
            current_predictions[index] = nfr_predictions[i]

        Y_training = transform_class_names_to_labels(training['_class_'])
        current_predictions = transform_class_names_to_labels(current_predictions)
        metrics_prf = precision_recall_fscore_support(Y_training, current_predictions, average='weighted')
        accuracy = accuracy_score(Y_training, current_predictions)

        current_results = (metrics_prf[0], metrics_prf[1], metrics_prf[2], accuracy)
        for i in range(len(training_results)):
            training_results[i] += current_results[i]

        # evaluation on test dataset
        current_predictions = model_binary.predict(X_test_binary)
        current_predictions = transform_labels_to_class_names(current_predictions)
        nfr_predictions_mask = current_predictions == 'NF'
        nfr_predictions_mask_indeces = np.where(current_predictions == 'NF')[0]

        two_step_samples = test[nfr_predictions_mask]
        X_two_step = two_step_samples.drop('_class_', axis=1)
        #Y_two_step = transform_class_names_to_labels(two_step_samples['_class_'])

        nfr_predictions = model_nfr.predict(X_two_step)
        nfr_predictions = transform_labels_to_class_names(nfr_predictions)
        for i in range(len(nfr_predictions_mask_indeces)):
            index = nfr_predictions_mask_indeces[i]
            current_predictions[index] = nfr_predictions[i]

        Y_test = transform_class_names_to_labels(test['_class_'])
        current_predictions = transform_class_names_to_labels(current_predictions)
        metrics_prf = precision_recall_fscore_support(Y_test, current_predictions, average='weighted')
        accuracy = accuracy_score(Y_test, current_predictions)

        current_results = (metrics_prf[0], metrics_prf[1], metrics_prf[2], accuracy)
        for i in range(len(test_results)):
            test_results[i] += current_results[i]

    for i in range(len(test_results)):
        training_results[i] /= K
        test_results[i] /= K
    
    line = 'Training: precision = {}; recall = {}; f1_score = {}; accuracy = {}'.format(training_results[0], training_results[1], training_results[2], training_results[3])
    output.write(line + '\n')
    output_info += '\n{}'.format(line)

    line = 'Test: precision = {}; recall = {}; f1_score = {}; accuracy = {}'.format(test_results[0], test_results[1], test_results[2], test_results[3])
    output.write(line + '\n')
    output_info += '\n{}'.format(line)

    output.close()
    return output_info

bow = evaluate_two_step_model('bow-12', 'two-step-bow-knn', 11, 1)
tfidf = evaluate_two_step_model('tfidf-12', 'two-step-tfidf-knn', 11, 1)
print('{}\n\n{}'.format(bow, tfidf))