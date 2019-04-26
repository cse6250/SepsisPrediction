import pickle

import numpy as np

from mydatasets import construct_features
from feature_importance import feature_importance
from mymodels import select_model, evaluate

PATH_TRAIN_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.train"
PATH_TRAIN_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.train"
PATH_VALID_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.validation"
PATH_VALID_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.validation"
PATH_TEST_SEQS = "./../../data/sepsis/processed_data/sepsis.seqs.test"
PATH_TEST_LABELS = "./../../data/sepsis/processed_data/sepsis.labels.test"

# loading data
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
y_train = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
y_valid = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
y_test = pickle.load(open(PATH_TEST_LABELS, 'rb'))

# construct features
X_train = construct_features(train_seqs)
X_valid = construct_features(valid_seqs)
X_test = construct_features(test_seqs)

X_train_valid = X_train + X_valid
y_train_valid = y_train + y_valid

# get the feature importance
feature_importance(X_train_valid, y_train_valid)

# prune the parameters by GridSearchCV 
select_model(X_train_valid, y_train_valid)

# evaluate the models on test set
result = evaluate(X_train_valid, X_test, y_train_valid, y_test)
