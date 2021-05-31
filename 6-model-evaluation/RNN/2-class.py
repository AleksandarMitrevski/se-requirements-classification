import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, GRU, LeakyReLU, Dense, Dropout, Embedding
from tensorflow.keras import backend as tf_backend
# fix for issue w/ CUDA v11.1 integration - https://github.com/tensorflow/tensorflow/issues/44567#issuecomment-728772416
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import os, sys
sys.path.insert(0, os.path.abspath('../..'))
from utils import utils

K = 10  # k-fold cross validation

class_names = ['F', 'NF']
class_names_map = { value: index for index, value in enumerate(class_names) }

transform_categorical_y_to_numeric = lambda x: class_names_map[x]
transform_numeric_y_to_categorical = lambda x: 1 if x >= 0.5 else 0 # threshold

def evaluate_binary_model(output_filename_base, mode, param_epochs, param_dropout, param_dense_dropout, param_dense_nodes, param_leaky_relu_alpha):
    output_info = '** {}-{} **'.format(output_filename_base, mode)
    print('\n' + output_info)
    output = open('../results/{}-{}.txt'.format(output_filename_base, mode), 'w')

    training_results_steps = np.zeros((K, 4))
    test_results_steps = np.zeros((K, 4))

    embeddings_matrix, max_sequence_length, dataset = utils.load_dataset_word2vec('./GoogleNews-vectors-negative300.bin', '../../2-preprocessing/output', '2-class')
    vocabulary_size = embeddings_matrix.shape[0] - 1 # embeddings_matrix has one extra row for unused zero index due to mask_zero in Embedding
    random_split = utils.cv_split(dataset, K)

    for i in range(K):
        print('\n   CV Split #{}'.format(i + 1))

        test = random_split[i]
        training_list = random_split[0:i] + random_split[i+1:K]
        training = pd.concat(training_list)

        X_train = training.drop('_class_', axis=1)
        Y_train = training['_class_'].apply(transform_categorical_y_to_numeric)

        tf_backend.clear_session() # clear keras session

        model = Sequential()
        model.add(
            Embedding(input_dim=vocabulary_size+1,  # one more than vocabulary size because mask_zero is True
                output_dim=utils.WORD2VEC_EMBEDDINGS_VECTOR_SIZE,
                weights=[embeddings_matrix],
                input_length=max_sequence_length,
                mask_zero=True,
                trainable=False))
        recurrent_layer = None
        if mode == 'lstm':
            recurrent_layer = LSTM(utils.WORD2VEC_EMBEDDINGS_VECTOR_SIZE, dropout=param_dropout)    # recurrent_dropout makes cuDNN usage (model training time optimization) infeasible, otherwise it would have been employed
        elif mode == 'gru':
            recurrent_layer = GRU(utils.WORD2VEC_EMBEDDINGS_VECTOR_SIZE, dropout=param_dropout)
        model.add(
            Bidirectional(recurrent_layer))
        model.add(Dropout(param_dense_dropout))
        model.add(
            Dense(param_dense_nodes, activation=LeakyReLU(alpha=param_leaky_relu_alpha)))
        model.add(
            Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')

        model.fit(x=X_train, y=Y_train, epochs=param_epochs, verbose=1)

        # evaluation on training dataset
        X_test = X_train
        training_prediction = model.predict(X_test)

        # evaluation on test dataset
        X_test = test.drop('_class_', axis=1)
        test_prediction = model.predict(X_test)

        # do the classification and evaluate performance
        Y_test = Y_train
        predictions = np.array([transform_numeric_y_to_categorical(prediction) for prediction in training_prediction])
        metrics_prf = precision_recall_fscore_support(Y_test, predictions, average='weighted')
        accuracy = accuracy_score(Y_test, predictions)
        current_results = (metrics_prf[0], metrics_prf[1], metrics_prf[2], accuracy)
        for j in range(training_results_steps.shape[1]):
            training_results_steps[i, j] = current_results[j]

        Y_test = test['_class_'].apply(transform_categorical_y_to_numeric)
        predictions = np.array([transform_numeric_y_to_categorical(prediction) for prediction in test_prediction])
        metrics_prf = precision_recall_fscore_support(Y_test, predictions, average='weighted')
        accuracy = accuracy_score(Y_test, predictions)
        current_results = (metrics_prf[0], metrics_prf[1], metrics_prf[2], accuracy)
        for j in range(test_results_steps.shape[1]):
            test_results_steps[i, j] = current_results[j]

    training_results_means = np.mean(training_results_steps, axis=0)
    test_results_means = np.mean(test_results_steps, axis=0)
    training_results_std = np.std(training_results_steps, axis=0, ddof=1)
    test_results_std = np.std(test_results_steps, axis=0, ddof=1)
    
    line = 'Training: precision = {} +/- {}; recall = {} +/- {}; f1_score = {} +/- {}; accuracy = {} +/- {}'.format(training_results_means[0], training_results_std[0], training_results_means[1], training_results_std[1], training_results_means[2], training_results_std[2], training_results_means[3], training_results_std[3])
    output.write(line + '\n')
    output_info += '\n{}'.format(line)

    line = 'Test: precision = {} +/- {}; recall = {} +/- {}; f1_score = {} +/- {}; accuracy = {} +/- {}'.format(test_results_means[0], test_results_std[0], test_results_means[1], test_results_std[1], test_results_means[2], test_results_std[2], test_results_means[3], test_results_std[3])
    output.write(line + '\n')
    output_info += '\n{}'.format(line)

    output.close()
    return output_info

lstm = evaluate_binary_model('2-w2v', 'lstm', 100, 0.9, 0.9, 10, 0.3)
gru = evaluate_binary_model('2-w2v', 'gru', 100, 0.9, 0.9, 10, 0.3)
print('{}\n\n{}'.format(lstm, gru))