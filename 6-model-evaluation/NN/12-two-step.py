import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as tf_backend

import os, sys
sys.path.insert(0, os.path.abspath("../.."))
from utils import utils

K = 10  # k-fold cross validation

class_names_binary = ['F', 'NF']
num_classes_binary = len(class_names_binary)
class_names_binary_map = { value: index for index, value in enumerate(class_names_binary) }
class_names_binary_map_reverse = { index: value for index, value in enumerate(class_names_binary) }
transform_categorical_y_to_numeric_binary = lambda x: class_names_binary_map[x]
transform_numeric_y_to_categorical_binary = lambda x: class_names_binary_map_reverse[x]
def transform_numeric_y_to_categorical_binary_np(arr):
    func = np.vectorize(transform_numeric_y_to_categorical_binary)
    return func(arr)

threshold_numeric_binary_y = lambda x: 1 if x >= 0.5 else 0 # threshold
def threshold_numeric_binary_y_np(arr):
    func = np.vectorize(threshold_numeric_binary_y)
    return func(arr)

class_names_nfr = ['A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
num_classes_nfr = len(class_names_nfr)
class_names_nfr_map = { value: index for index, value in enumerate(class_names_nfr) }
class_names_nfr_map_reverse = { index: value for index, value in enumerate(class_names_nfr) }
transform_categorical_y_to_numeric_nfr = lambda x: class_names_nfr_map[x]
transform_numeric_y_to_categorical_nfr = lambda x: class_names_nfr_map_reverse[x]
def transform_numeric_y_to_categorical_nfr_np(arr):
    func = np.vectorize(transform_numeric_y_to_categorical_nfr)
    return func(arr)

class_names_total = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
num_classes_total = len(class_names_total)
class_names_total_map = { value: index for index, value in enumerate(class_names_total) }
class_names_total_map_reverse = { index: value for index, value in enumerate(class_names_total) }
transform_categorical_y_to_numeric_total = lambda x: class_names_total_map[x]
def transform_categorical_y_to_numeric_total_np(arr):
    func = np.vectorize(transform_categorical_y_to_numeric_total)
    return func(arr)
transform_numeric_y_to_categorical_total = lambda x: class_names_total_map_reverse[x]
def transform_numeric_y_to_categorical_total_np(arr):
    func = np.vectorize(transform_numeric_y_to_categorical_total)
    return func(arr)

def evaluate_two_step_model(dataset, output_filename, param_epochs_binary, param_nodes_per_layer_binary, param_hidden_layers_binary, param_dropout_binary, param_epochs_nfr, param_nodes_per_layer_nfr, param_hidden_layers_nfr, param_dropout_nfr):
    output_info = '** {} **'.format(dataset)
    output = open('../results/{}.txt'.format(output_filename), "w")

    training_results = [0, 0, 0, 0]
    test_results = [0, 0, 0, 0]

    df = utils.load_dataset('../../4-feature-selection/output', dataset, True)
    random_split = utils.cv_split(df, K)

    for j in range(K):
        print('\n   CV Split #{}'.format(j + 1))
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
        Y_train_binary = binary_training['_class_'].apply(transform_categorical_y_to_numeric_binary)
        X_test_binary = binary_test.drop('_class_', axis=1)
        #Y_test_binary = transform_class_names_to_labels(binary_test['_class_'])

        X_train_nfr = nfr_training.drop('_class_', axis=1)
        Y_train_nfr = nfr_training['_class_'].apply(transform_categorical_y_to_numeric_nfr)
        Y_train_matrix_nfr = to_categorical(Y_train_nfr)
        
        tf_backend.clear_session() # clear keras session

        print('\tTraining NN - binary')
        model_binary = Sequential()
        model_binary.add(Dropout(param_dropout_binary))
        for _ in range(param_hidden_layers_binary):
            model_binary.add(Dense(param_nodes_per_layer_binary, activation='relu'))
        model_binary.add(Dense(1, activation='sigmoid'))
        model_binary.compile(loss='binary_crossentropy', optimizer='adam')

        model_binary.fit(x=X_train_binary, y=Y_train_binary, epochs=param_epochs_binary, verbose=0)

        print('\tTraining NN - NFR')
        model_nfr = Sequential()
        model_nfr.add(Dropout(param_dropout_nfr))
        for _ in range(param_hidden_layers_nfr):
            model_nfr.add(Dense(param_nodes_per_layer_nfr, activation='relu'))
        model_nfr.add(Dense(num_classes_nfr, activation='softmax'))
        model_nfr.compile(loss='categorical_crossentropy', optimizer='adam')

        model_nfr.fit(x=X_train_nfr, y=Y_train_matrix_nfr, epochs=param_epochs_nfr, verbose=0)

        # evaluation on training set
        current_predictions = model_binary.predict(X_train_binary)
        current_predictions = threshold_numeric_binary_y_np(current_predictions)
        current_predictions = transform_numeric_y_to_categorical_binary_np(current_predictions)
        nfr_predictions_mask = current_predictions == 'NF'
        nfr_predictions_mask_indeces = np.where(current_predictions == 'NF')[0]

        two_step_samples = training[nfr_predictions_mask]
        X_two_step = two_step_samples.drop('_class_', axis=1)
        #Y_two_step = transform_categorical_y_to_numeric_total_np(two_step_samples['_class_'])  # there would be problems if this is used as is, as total != nfr, but total must be used

        prediction = model_nfr.predict(X_two_step)
        nfr_predictions = np.zeros(shape=(prediction.shape[0], 1))
        for i in range(prediction.shape[0]):
            max_index = -1
            max_value = -1
            for k in range(prediction.shape[1]):
                if prediction[i, k] > max_value:
                    max_index = k
                    max_value = prediction[i, k]
            nfr_predictions[i] = max_index
        nfr_predictions = transform_numeric_y_to_categorical_nfr_np(nfr_predictions)

        for i in range(len(nfr_predictions_mask_indeces)):
            index = nfr_predictions_mask_indeces[i]
            current_predictions[index] = nfr_predictions[i]

        Y_training = training['_class_'].apply(transform_categorical_y_to_numeric_total)
        current_predictions = transform_categorical_y_to_numeric_total_np(current_predictions)
        metrics_prf = precision_recall_fscore_support(Y_training, current_predictions, average='weighted')
        accuracy = accuracy_score(Y_training, current_predictions)

        current_results = (metrics_prf[0], metrics_prf[1], metrics_prf[2], accuracy)
        for i in range(len(training_results)):
            training_results[i] += current_results[i]

        # evaluation on test dataset
        current_predictions = model_binary.predict(X_test_binary)
        current_predictions = threshold_numeric_binary_y_np(current_predictions)
        current_predictions = transform_numeric_y_to_categorical_binary_np(current_predictions)
        nfr_predictions_mask = current_predictions == 'NF'
        nfr_predictions_mask_indeces = np.where(current_predictions == 'NF')[0]

        two_step_samples = test[nfr_predictions_mask]
        X_two_step = two_step_samples.drop('_class_', axis=1)
        #Y_two_step = transform_categorical_y_to_numeric_total_np(two_step_samples['_class_'])  # see comment on same line in training

        prediction = model_nfr.predict(X_two_step)
        nfr_predictions = np.zeros(shape=(prediction.shape[0], 1))
        for i in range(prediction.shape[0]):
            max_index = -1
            max_value = -1
            for k in range(prediction.shape[1]):
                if prediction[i, k] > max_value:
                    max_index = k
                    max_value = prediction[i, k]
            nfr_predictions[i] = max_index
        nfr_predictions = transform_numeric_y_to_categorical_nfr_np(nfr_predictions)

        for i in range(len(nfr_predictions_mask_indeces)):
            index = nfr_predictions_mask_indeces[i]
            current_predictions[index] = nfr_predictions[i]

        Y_test = test['_class_'].apply(transform_categorical_y_to_numeric_total)
        current_predictions = transform_categorical_y_to_numeric_total_np(current_predictions)
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

bow = evaluate_two_step_model('bow-12', 'two-step-bow-nn', 50, 40, 1, 2/3, 40, 100, 1, 1/3)
tfidf = evaluate_two_step_model('tfidf-12', 'two-step-tfidf-nn', 50, 40, 1, 2/3, 40, 100, 1, 1/3)
print('{}\n\n{}'.format(bow, tfidf))