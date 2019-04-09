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

#---get feature lenght---
file = open('training/p00001.psv',"r")
for line in file:
    data = line.split("|")
    feature_length = len(data)-1
#    file.close()

#---get training lenght---
path = "training/" 
files= os.listdir(path) 
dataset_num = 0
for f in files:
    dataset_num = dataset_num + 1

def original_get_data(dataset_num, feature_length):
    allX = np.zeros((dataset_num, feature_length))
    allY = np.zeros(dataset_num, dtype=int)
    
    #---build training X and Y---
    num = 0
    for f in files:
    #    with open('data/'+ file) as f:
    
        file = open('training/'+ f,"r")
        train_num = 0
        for line in file:
            data = line.split("|")
            feature_length = len(data)-1
            train_num = train_num  + 1
            
        X = np.zeros((train_num-1, feature_length))
        Y = np.zeros(train_num-1)
    #    file.close()
        
        i = 0
        file = open('training/'+ f,"r")
        for line in file:
            if(i!=0):
                data = line.split("|")
                features = data[0:feature_length]
                for f in range(0,len(features)):
                    if (features[f] == 'NaN'):
                        features[f] = 0
                    else:
                        features[f] = float(features[f])
                X[i-1,:] = features
                label = int(data[-1].rstrip("\n"))
                Y[i-1] = label
                
            i = i+1
        
        allX[num,:] = X.mean(0)
        allY[num] = max(Y)
        num = num + 1
    #    file.close()       
    feature_mean = allX.mean(0)
    return allX, allY, feature_mean

allX, allY, feature_mean = original_get_data(dataset_num, feature_length)

def replace_get_data(dataset_num, feature_length, feature_mean):    
#---take the mean for each feature to replace 0---
    allX_mean_replace = np.zeros((dataset_num, feature_length))
    allY_mean_replace = np.zeros(dataset_num, dtype=int)
    
    #---build training X and Y---
    num = 0
    for f in files:
    #    with open('data/'+ file) as f:
    
        file = open('training/'+ f,"r")
        train_num = 0
        for line in file:
            data = line.split("|")
            feature_length = len(data)-1
            train_num = train_num  + 1
            
        X = np.zeros((train_num-1, feature_length))
        Y = np.zeros(train_num-1)
    #    file.close()
        
        i = 0
        file = open('training/'+ f,"r")
        for line in file:
            if(i!=0):
                data = line.split("|")
                features = data[0:feature_length]
                for f in range(0,len(features)):
                    if (features[f] == 'NaN'):
                        features[f] = feature_mean[f]
                    else:
                        features[f] = float(features[f])
                X[i-1,:] = features
                label = int(data[-1].rstrip("\n"))
                Y[i-1] = label
                
            i = i+1
        
        allX_mean_replace[num,:] = X.mean(0)
        allY_mean_replace[num] = max(Y)
        num = num + 1
    #    file.close()
    return allX_mean_replace, allY_mean_replace
  
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
        RandomForestClassifier(max_depth=5, n_estimators=50),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
#        GaussianNB(),
#        QuadraticDiscriminantAnalysis()
]
    
    print("Clf Name, Score, Auc, CM:")
    for name, clf in zip(names, classifiers):
    #    clf.fit(X_train, y_train)
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        score = clf.score(X_test, y_test)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        print(name, score, auc, '\n', cm)
        
#---train by each time and then predict the last 4 hours---
        
def get_time_data():
    trainX_mean_replace_time = []
    trainY_mean_replace_time = []
    predX_mean_replace_time = []
    predY_mean_replace_time = []
    
    for f in files:
    #    with open('data/'+ file) as f:
    
        file = open('training/'+ f,"r")
        train_num = 0
        for line in file:
            data = line.split("|")
            feature_length = len(data)-1
            train_num = train_num  + 1
            
        X = np.zeros((train_num-1, feature_length))
        Y = np.zeros(train_num-1)
    #    file.close()
        
        i = 0
        file = open('training/'+ f,"r")
        for line in file:
            if(i!=0): #except for the head
                data = line.split("|")
                features = data[0:feature_length]
                for f in range(0,len(features)):
                    if (features[f] == 'NaN'):
                        features[f] = feature_mean[f]
                    else:
                        features[f] = float(features[f])
                X[i-1,:] = features
                label = int(data[-1].rstrip("\n"))
                Y[i-1] = label
            
            i = i+1
        
        trainX_mean_replace_time.extend(X[0:train_num-5])
        predX_mean_replace_time.extend(X[train_num-5:train_num-1])
        trainY_mean_replace_time.extend(Y[0:train_num-5])
        predY_mean_replace_time.extend(Y[train_num-5:train_num-1])
        
    trainX_mean_replace_time = np.asarray(trainX_mean_replace_time)
    predX_mean_replace_time = np.asarray(predX_mean_replace_time)
    trainY_mean_replace_time = np.asarray(trainY_mean_replace_time, dtype = int)
    predY_mean_replace_time = np.asarray(predY_mean_replace_time, dtype = int)
    
    return trainX_mean_replace_time, predX_mean_replace_time, trainY_mean_replace_time, predY_mean_replace_time
 
def feature_importance(allX, allY):
    import matplotlib.pyplot as plt
    from sklearn.ensemble import ExtraTreesClassifier   
    
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)
    
    forest.fit(allX, allY)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(allX.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(allX.shape[1]), importances[indices],
           color="r", yerr=std[indices], align="center")
    plt.xticks(range(allX.shape[1]), indices)
    plt.xlim([-1, allX.shape[1]])
    plt.show()
    
#===========main==============
print("----------original----------")
X_train, X_test, y_train, y_test = train_test_split(allX, allY, test_size=0.5, random_state=42)
#MLtraining(X_train, X_test, y_train, y_test)

print("----------replace with mean----------")
allX_mean_replace, allY_mean_replace = replace_get_data(dataset_num, feature_length,feature_mean)
X_train, X_test, y_train, y_test = train_test_split(allX_mean_replace, allY_mean_replace, test_size=0.5, random_state=42)
#MLtraining(X_train, X_test, y_train, y_test)
#feature_importance(allX_mean_replace, allY_mean_replace)

print("----------4 hours prediction----------")
trainX_mean_replace_time, predX_mean_replace_time, trainY_mean_replace_time, predY_mean_replace_time = get_time_data()
#MLtraining(trainX_mean_replace_time, predX_mean_replace_time, trainY_mean_replace_time, predY_mean_replace_time)
#feature_importance(trainX_mean_replace_time, trainY_mean_replace_time)

newtrainX = trainX_mean_replace_time[:,[0,1,3,4,5,6,34,38,39]]
newpredX = predX_mean_replace_time[:,[0,1,3,4,5,6,34,38,39]]

#MLtraining(newtrainX, newpredX, trainY_mean_replace_time, predY_mean_replace_time)
print("----------feature selection------------")
newX = allX[:,[7,39]]
X_train, X_test, y_train, y_test = train_test_split(newX, allY, test_size=0.5, random_state=42)
#MLtraining(X_train, X_test, y_train, y_test)

def get_important_feature():
    file = open('training/p00001.psv',"r")
    for line in file:
        data = line.split("|")
        feature_name = data 
        break
    
    l = [39, 38, 34, 0, 3, 6, 4, 5, 1]
    topfeature = []
    for i in range(len(l)):
        topfeature.append(feature_name[l[i]])
    return topfeature

    
#============generate balanced spesis samples==============
print("----------pos neg sampling---------------")    
pos_ids = np.where(trainY_mean_replace_time==0)[0]
neg_ids = np.where(trainY_mean_replace_time==1)[0]
newly_number = len(neg_ids)
new_pos_ids = np.random.choice(pos_ids, newly_number*20)
new_trainX_hour_pos = trainX_mean_replace_time[new_pos_ids] 
new_trainX_hour_neg = trainX_mean_replace_time[neg_ids]
new_trainX_hour_neg = np.concatenate((new_trainX_hour_neg, new_trainX_hour_neg, new_trainX_hour_neg, new_trainX_hour_neg, new_trainX_hour_neg)) #copy spesis==1
new_trainX_hour = np.concatenate((new_trainX_hour_pos, new_trainX_hour_neg))
 
new_trainY_hour_pos = np.zeros(len(new_trainX_hour_pos))
new_trainY_hour_neg = np.ones(len(new_trainX_hour_neg))
new_trainY_hour = np.concatenate((new_trainY_hour_pos, new_trainY_hour_neg))

#MLtraining(new_trainX_hour, predX_mean_replace_time, new_trainY_hour, predY_mean_replace_time)
MLtraining(new_trainX_hour[:,[0,1,3,4,5,6,34,38,39]], predX_mean_replace_time[:,[0,1,3,4,5,6,34,38,39]], new_trainY_hour, predY_mean_replace_time)

#print(len(np.where(predY_mean_replace_time==0)[0])/len(np.where(predY_mean_replace_time==1)[0]))