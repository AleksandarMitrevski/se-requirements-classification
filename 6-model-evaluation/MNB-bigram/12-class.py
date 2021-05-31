import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

import os, sys
sys.path.insert(0, os.path.abspath("../.."))
from utils import utils

K = 10  # k-fold cross validation

class_names = ['F', 'A', 'L', 'LF', 'MN', 'O', 'PE', 'SC', 'SE', 'US', 'FT', 'PO']
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

output_info = '** 12-bigram **'
output = open('../results/12-bigram-mnb.txt', "w")

training_results = [0, 0, 0, 0]
test_results = [0, 0, 0, 0]

df = utils.load_bigram_dataset('../../', False)
random_split = utils.cv_split(df, K)

for j in range(K):
    test = random_split[j]
    training_list = random_split[0:j] + random_split[j+1:K]
    training = pd.concat(training_list)

    X_train = training.drop('_class_', axis=1)
    Y_train = label_encoder.transform(training['_class_'])
    base_model = MultinomialNB(alpha=0.1)
    model = OneVsRestClassifier(estimator=base_model, n_jobs=-1) # -1 means 'use all processors'
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
