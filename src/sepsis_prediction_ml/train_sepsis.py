import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier   
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
import matplotlib.pyplot as plt

RANDOM_STATE = 41
PATH_TRAIN = "./../../data/sepsis/train/train_sample_cleaned_pivoted_vital.csv"
PATH_VALIDATION = "./../../data/sepsis/validation/valid_sample_cleaned_pivoted_vital.csv"
PATH_TEST = "./../../data/sepsis/test/test_sample_cleaned_pivoted_vital.csv"

def evaluate(X_train, X_test, y_train, y_test):
    names = ["Decision Tree", "Random Forest", "AdaBoost", "Neural Net"]
    
    classifiers = [
        DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE), 
        RandomForestClassifier(max_depth=8, n_estimators = 100, random_state=RANDOM_STATE),
        MLPClassifier(alpha = 0.01, random_state=RANDOM_STATE),
        AdaBoostClassifier(n_estimators = 100, random_state=RANDOM_STATE),
    ]
    
    result = []
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)  
        roc_auc = roc_auc_score(y_test, y_prob)
        mcc = matthews_corrcoef(y_test, y_pred)
        result.append([precision, recall, f1, mcc, roc_auc])
        print("\nEvaluation metrics on " + name + ': \t')
        print('Test Accuracy: ' + str(accuracy) + '\t') 
        print('Test Precision: ' + str(precision) + '\t')
        print('Test Recall: ' + str(recall) + '\t')
        print('Test F1-score: ' + str(f1) + '\t')
        print('Test ROC-AUC: ' + str(roc_auc) + '\t')
        print('Test MCC: ' + str(mcc) + '\t')
    return np.array(result)

#============part 1: take the mean, and then give a label for this patient=======
#---take the last three/four hours label as the prediction goal
def get_data_mean(data, prediction_window):
    group = data.groupby(data['icustay_id'])
    mean_data = group.mean()
    patients_num = len(mean_data)
    new_data_list = list(group)
    prediction_window = 3
    
    X = []
    Y = []
    
    for i in range(patients_num):
        tempY = new_data_list[i][1]['sepsis_label'].values.tolist()[-prediction_window:] 
        # the last prediction_window hours
        tempX = np.array(new_data_list[i][1][feature_name[2:-1]].values.tolist()[0:-prediction_window])
        # the first (lenght-prediction_window) hours as the input feature
        if(len(tempX>0)):
            X.append(tempX.mean(0))
            Y.append(max(tempY))
    
    X = np.array(X)
    Y = np.array(Y)  
    print('load data finish!')
    return X,Y

#===============part 2: each three has a label
    
def get_data_window(data):
    group = data.groupby(data['icustay_id'])
    mean_data = group.mean()
    patients_num = len(mean_data)
    new_data_list = list(group)
    prediction_window = 2
    
    X = []
    Y = []
    
    for i in range(patients_num):
        for j in range(round(patients_num/3)):
            tempY = new_data_list[i][1]['sepsis_label'].values.tolist()[j:j+prediction_window] 
        # the last prediction_window hours
            tempX = np.array(new_data_list[i][1][feature_name[2:-1]].values.tolist()[j:j+prediction_window])
        # the first (lenght-prediction_window) hours as the input feature
            if(len(tempX>0)):
                X.extend(tempX)
                Y.extend(tempY)
    
    X = np.array(X)
    Y = np.array(Y)  
    print('load data finish!')
    return X,Y

#-------------------------------main-----------------------------------     
train_data = pd.read_csv(PATH_TRAIN)
validation_data = pd.read_csv(PATH_VALIDATION)
test_data = pd.read_csv(PATH_TEST)
prediction_window = 6

feature_name = train_data.columns.values.tolist()

X_train, y_train = get_data_mean(train_data, prediction_window)
X_test, y_test = get_data_mean(test_data, prediction_window)
result = evaluate(X_train, X_test, y_train, y_test)
