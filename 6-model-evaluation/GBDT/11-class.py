import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import os, sys
sys.path.insert(0, os.path.abspath("../.."))
from utils import utils

K = 10  # k-fold cross validation

class_names = ['A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

SELECTED_ESTIMATORS = 200
SELECTED_LEARNING_RATE = 0.05

def evaluate_multiclass_model(dataset, output_filename):
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

        X_train = training.drop('_class_', axis=1)
        Y_train = label_encoder.transform(training['_class_'])
        model = GradientBoostingClassifier(n_estimators=SELECTED_ESTIMATORS, learning_rate=SELECTED_LEARNING_RATE, max_features='sqrt')
        model.fit(X_train, Y_train)

        # evaluation on training dataset
        X_test = X_train
        Y_test = Y_train
        current_results = utils.estimate_model_performance(model, X_test, Y_test)
        for i in range(len(training_results)):
            training_results[i] += current_results[i]

        # evaluation on test dataset
        X_test = test.drop('_class_', axis=1)
        Y_test = label_encoder.transform(test['_class_'])
        current_results = utils.estimate_model_performance(model, X_test, Y_test)
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

bow = evaluate_multiclass_model('bow-11', '11-bow-gbdt')
tfidf = evaluate_multiclass_model('tfidf-11', '11-tfidf-gbdt')
print('{}\n\n{}'.format(bow, tfidf))