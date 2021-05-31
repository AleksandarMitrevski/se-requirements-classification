import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as tf_backend

import os, sys
sys.path.insert(0, os.path.abspath('../..'))
from utils import utils

K = 10  # k-fold cross validation

class_names = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
num_classes = len(class_names)
class_names_map = { value: index for index, value in enumerate(class_names) }

transform_categorical_y_to_numeric = lambda x: class_names_map[x]

def evaluate_multiclass_model(dataset, output_filename, param_epochs, param_nodes_per_layer, param_hidden_layers, param_ensemble_size, param_dropout):
    output_info = '** {} **'.format(dataset)
    print('\n' + output_info)
    output = open('../results/{}.txt'.format(output_filename), 'w')

    training_results = [0, 0, 0, 0]
    test_results = [0, 0, 0, 0]

    df = utils.load_dataset('../../4-feature-selection/output', dataset, True)
    random_split = utils.cv_split(df, K)

    for j in range(K):
        print('\n   CV Split #{}'.format(j + 1))
        training_predictions = [None] * param_ensemble_size
        test_predictions = [None] * param_ensemble_size

        test = random_split[j]
        training_list = random_split[0:j] + random_split[j+1:K]
        training = pd.concat(training_list)

        X_train = training.drop('_class_', axis=1)
        Y_train = training['_class_'].apply(transform_categorical_y_to_numeric)
        Y_train_matrix = to_categorical(Y_train)

        for i in range(param_ensemble_size):
            tf_backend.clear_session() # clear keras session

            print('\tTraining NN #{}'.format(i + 1))
            model = Sequential()
            model.add(Dropout(param_dropout))
            for _ in range(param_hidden_layers):
                model.add(Dense(param_nodes_per_layer, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam')

            model.fit(x=X_train, y=Y_train_matrix, epochs=param_epochs, verbose=0)

            # evaluation on training dataset
            X_test = X_train
            training_predictions[i] = model.predict(X_test)

            # evaluation on test dataset
            X_test = test.drop('_class_', axis=1)
            test_predictions[i] = model.predict(X_test)

        # average the class probabilities of the ensemble, then select the class with the maximum probability
        training_predictions_average = np.zeros(training_predictions[0].shape)
        for i in range(training_predictions_average.shape[0]):
            for p in range(training_predictions_average.shape[1]):
                for k in range(param_ensemble_size):
                    training_predictions_average[i, p] += training_predictions[k][i, p]
                training_predictions_average[i, p] /= param_ensemble_size
        training_prediction = np.zeros(shape=(training_predictions_average.shape[0], 1))
        for i in range(training_predictions_average.shape[0]):
            max_index = -1
            max_value = -1
            for k in range(training_predictions_average.shape[1]):
                if training_predictions_average[i, k] > max_value:
                    max_index = k
                    max_value = training_predictions_average[i, k]
            training_prediction[i] = max_index

        test_predictions_average = np.zeros(test_predictions[0].shape)
        for i in range(test_predictions_average.shape[0]):
            for p in range(test_predictions_average.shape[1]):
                for k in range(param_ensemble_size):
                    test_predictions_average[i, p] += test_predictions[k][i, p]
                test_predictions_average[i, p] /= param_ensemble_size
        test_prediction = np.zeros(shape=(test_predictions_average.shape[0], 1))
        for i in range(test_predictions_average.shape[0]):
            max_index = -1
            max_value = -1
            for k in range(test_predictions_average.shape[1]):
                if test_predictions_average[i, k] > max_value:
                    max_index = k
                    max_value = test_predictions_average[i, k]
            test_prediction[i] = max_index

        # training_prediction and test_prediction are now compatible with Y transformed by transform_categorical_y_to_numeric

        # do the classification and evaluate performance
        Y_test = Y_train
        metrics_prf = precision_recall_fscore_support(Y_test, training_prediction, average='weighted')
        accuracy = accuracy_score(Y_test, training_prediction)
        current_results = (metrics_prf[0], metrics_prf[1], metrics_prf[2], accuracy)
        for i in range(len(training_results)):
            training_results[i] += current_results[i]

        Y_test = test['_class_'].apply(transform_categorical_y_to_numeric)
        metrics_prf = precision_recall_fscore_support(Y_test, test_prediction, average='weighted')
        accuracy = accuracy_score(Y_test, test_prediction)
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

bow = evaluate_multiclass_model('bow-12', '12-bow-nn', 40, 100, 1, 1, 1/3)
tfidf = evaluate_multiclass_model('tfidf-12', '12-tfidf-nn', 40, 100, 1, 1, 1/3)
print('{}\n\n{}'.format(bow, tfidf))