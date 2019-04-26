import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier   

RANDOM_STATE = 41

def feature_importance(X, y):

    forest = ExtraTreesClassifier(n_estimators=250, random_state=RANDOM_STATE)
    forest.fit(X, y)
    importances = forest.feature_importances_
    
    # Plot the feature importances of the forest
    feature_name = np.array(['heartrate', 'sysbp', 'diasbp' , 'meanbp' , 'resprate', 'tempc', 'spo2', 'glucose', 'age', 'gender'])

    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    X = np.array(X)

    fig = plt.figure(figsize=(15,6))
    plt.bar(range(X.shape[1]), importances[indices],
           color="y", yerr=std[indices], align="center", width = 0.6)
    plt.xticks(range(X.shape[1]), feature_name[indices]) 
    plt.xlim([-1, X.shape[1]])
    fig.savefig('./../../out/img/feature_importance.png')