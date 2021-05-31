import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

import os, sys
sys.path.insert(0, os.path.abspath("../.."))
from utils import utils

class_names = ['F', 'NF']
label_encoder = LabelEncoder()
label_encoder.fit(class_names)

def model_select(dataset):
    output_info = '** {} **\n'.format(dataset)
    df = utils.load_dataset('../../4-feature-selection/output', dataset, True)

    K = 10  # k-fold cross validation
    hyperparam1_candidates = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]  # this is too many models; consider limiting search to desired pairs
    hyperparam2_candidates = [0.25, 0.1, 0.05, 0.01, 0.001]
    i = 0
    for hyperparam1_candidate in hyperparam1_candidates:
        for hyperparam2_candidate in hyperparam2_candidates:
            random_split = utils.cv_split(df, K)
            current_f1 = 0
            for j in range(K):
                test = random_split[j]
                training_list = random_split[0:j] + random_split[j+1:K]
                training = pd.concat(training_list)

                X_train = training.drop('_class_', axis=1)
                Y_train = label_encoder.transform(training['_class_'])
                X_test = test.drop('_class_', axis=1)
                Y_test = label_encoder.transform(test['_class_'])
                model = GradientBoostingClassifier(n_estimators=hyperparam1_candidate, learning_rate=hyperparam2_candidate, max_features='sqrt')
                model.fit(X_train, Y_train)
                results = utils.estimate_model_performance(model, X_test, Y_test)
                
                current_f1 += results[0]

            current_f1 /= K
            output_info += 'step {}: {}, {} - {}\n'.format(i + 1, hyperparam1_candidate, hyperparam2_candidate, current_f1)
            i += 1
        
    return output_info

bow = model_select('bow-2')
tfidf = model_select('tfidf-2')
print('{}\n{}'.format(bow, tfidf))