#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 17:16:27 2019

@author: renwendi
"""

import os
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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score


file = open('training/p00001.psv',"r")
for line in file:
    data = line.split("|")
    feature_length = len(data)-1

path = "training/" 
files= os.listdir(path) 
dataset_num = 0
for f in files:
    dataset_num = dataset_num + 1

allX = np.zeros((dataset_num, feature_length))
allY = np.zeros(dataset_num, dtype=int)
    
path = "training/" 
num = 0
files= os.listdir(path) 
for f in files:
#    with open('data/'+ file) as f:

    file = open('training/'+ f,"r")

    train_num = 0
    for line in file:
        data = line.split("|")
        feature_length = len(data)-1
        train_num = train_num  + 1
#    print(train_num)
        
    X = np.zeros((train_num-1, feature_length))
    Y = np.zeros(train_num-1)
    
    i = 0
    file = open('training/'+ f,"r")
    for line in file:
        if(i!=0):
      #Let's split the line into an array called "fields" using the ";" as a separator:
    #      print(line)
            data = line.split("|")
            data_length = len(data)
            features = data[0:data_length-1]
            for f in range(0,len(features)):
                if (features[f] == 'NaN'):
    #              print(features[f])
                    features[f] = 0
                else:
    #              print(features[f])
                    features[f] = float(features[f])
    #        print(features)
            X[i-1,:] = features
            label = int(data[-1].rstrip("\n"))
            Y[i-1] = label
            
        i = i+1
    
    allX[num,:] = X.mean(0)
    allY[num] = max(Y)
    num = num + 1
    
X_train, X_test, y_train, y_test = train_test_split(allX, allY, test_size=0.5, random_state=42)

from sklearn import svm
classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)
score = classifier.score(X_test, y_test)
cm = confusion_matrix(y_test, y_pred)


names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

for name, clf in zip(names, classifiers):
#    clf.fit(X_train, y_train)
    y_pred = clf.fit(X_train, y_train).predict(X_test)
    score = clf.score(X_test, y_test)
    cm = confusion_matrix(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    print(name, score, auc, '\n', cm)
    