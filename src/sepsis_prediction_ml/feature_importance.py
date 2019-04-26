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
    
    forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
    
    forest.fit(allX, allY)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    #print(feature_name[indices])
    
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