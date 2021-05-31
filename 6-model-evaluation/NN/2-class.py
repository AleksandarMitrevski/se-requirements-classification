import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import backend as tf_backend

import os, sys
sys.path.insert(0, os.path.abspath('../..'))
from utils import utils

K = 10  # k-fold cross validation

class_names = ['F', 'NF']
class_names_map = { value: index for index, value in enumerate(class_names) }

transform_categorical_y_to_numeric = lambda x: class_names_map[x]
transform_numeric_y_to_categorical = lambda x: 1 if x >= 0.5 else 0 # threshold

def evaluate_binary_model(dataset, output_filename, param_epochs, param_nodes_per_layer, param_hidden_layers, param_ensemble_size, param_dropout):
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

        for i in range(param_ensemble_size):
            tf_backend.clear_session() # clear keras session

            print('\tTraining NN #{}'.format(i + 1))
            model = Sequential()
            model.add(Dropout(param_dropout))
            for _ in range(param_hidden_layers):
                model.add(Dense(param_nodes_per_layer, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam')

            model.fit(x=X_train, y=Y_train, epochs=param_epochs, verbose=0)

            # evaluation on training dataset
            X_test = X_train
            training_predictions[i] = model.predict(X_test)

            # evaluation on test dataset
            X_test = test.drop('_class_', axis=1)
            test_predictions[i] = model.predict(X_test)

        # average the class probabilities of the ensemble
        training_prediction = np.zeros(training_predictions[0].shape)
        for i in range(training_prediction.shape[0]):
            for k in range(param_ensemble_size):
                training_prediction[i] += training_predictions[k][i]
            training_prediction[i] /= param_ensemble_size

        test_prediction = np.zeros(test_predictions[0].shape)
        for i in range(test_prediction.shape[0]):
            for k in range(param_ensemble_size):
                test_prediction[i] += test_predictions[k][i]
            test_prediction[i] /= param_ensemble_size

        # do the classification and evaluate performance
        Y_test = Y_train
        predictions = np.array([transform_numeric_y_to_categorical(prediction) for prediction in training_prediction])
        metrics_prf = precision_recall_fscore_support(Y_test, predictions, average='weighted')
        accuracy = accuracy_score(Y_test, predictions)
        current_results = (metrics_prf[0], metrics_prf[1], metrics_prf[2], accuracy)
        for i in range(len(training_results)):
            training_results[i] += current_results[i]

        Y_test = test['_class_'].apply(transform_categorical_y_to_numeric)
        predictions = np.array([transform_numeric_y_to_categorical(prediction) for prediction in test_prediction])
        metrics_prf = precision_recall_fscore_support(Y_test, predictions, average='weighted')
        accuracy = accuracy_score(Y_test, predictions)
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

bow = evaluate_binary_model('bow-2', '2-bow-nn', 50, 40, 1, 1, 2/3)
tfidf = evaluate_binary_model('tfidf-2', '2-tfidf-nn', 50, 40, 1, 1, 2/3)
print('{}\n\n{}'.format(bow, tfidf))