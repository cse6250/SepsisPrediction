#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:22:34 2019

@author: renwendi
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, normalize):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cmap=plt.cm.Blues

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = [0, 1]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")
    fig.tight_layout()
    return ax

def MLtraining(X_train, X_test, y_train, y_test):
#    names = ["Nearest Neighbors", "Gaussian Process",
#             "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
#             "Naive Bayes", "QDA"]
    names = ["Random Forest", "Neural Net", "AdaBoost"]
    
    classifiers = [
        KNeighborsClassifier(3),
#        SVC(kernel="linear", C=0.025),
#        SVC(gamma=2, C=1),
#        GaussianProcessClassifier(1.0 * RBF(1.0)),
#        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=8, n_estimators = 100),
        MLPClassifier(alpha = 0.01),
        AdaBoostClassifier(n_estimators = 100),
#        GaussianNB(),
#        QuadraticDiscriminantAnalysis()
]
    
    
    print("Clf Name, Score, Auc, CM:")
    result = []
    for name, clf in zip(names, classifiers):
    #    clf.fit(X_train, y_train)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
       
        plot_confusion_matrix(y_test, y_pred, 1)
        
        precision = precision_score(y_test, y_pred)
        recall = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        score = clf.score(X_test, y_test)       
        auc = roc_auc_score(y_test, y_pred)
        
        result.append([precision, recall, f1, score, auc])
        print(name, precision, recall, f1, score, auc) 
    return np.array(result)
#len(np.where(label==1)[0])

#data.head(100)

#data = data.loc[0:5000000]  # for test


#============part 1: take the mean, and then give a label for this patient=======
#---take the last three/four hours label as the prediction goal
def get_data_mean(data):
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
    prediction_window = 3
    
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

#data = pd.read_csv('cleaned_pivoted_vital.csv') 
data = pd.read_csv('sample_cleaned_pivoted_vital.csv')
feature_name = data.columns.values.tolist()

X,Y = get_data_mean(data)
#X,Y = get_data_window(data)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
#result = MLtraining(X_train, X_test, y_train, y_test)

def feature_importance(allX, allY, feature_name):
    feature_name = np.array(feature_name)
        
    SMALL_SIZE = 14
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 14
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    from sklearn.ensemble import ExtraTreesClassifier   
    
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    
    forest.fit(allX, allY)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
#    print(feature_name[indices])
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(allX.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure(figsize=(15,6))
    plt.bar(range(allX.shape[1]), importances[indices],
           color="y", yerr=std[indices], align="center", width = 0.6)
    plt.xticks(range(allX.shape[1]), feature_name[2:-1][indices]) 
    plt.xlim([-1, allX.shape[1]])
    plt.show()
    return indices, importances
     
indices, importances = feature_importance(X, Y, feature_name)
