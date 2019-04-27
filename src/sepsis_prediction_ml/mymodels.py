from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

RANDOM_STATE = 41

def select_model(X, y):
    names = ["MyDT", "MyRF", "MyAdaBoost", "MyMLP"]

    estimators = [
        DecisionTreeClassifier(random_state=RANDOM_STATE), 
        RandomForestClassifier(random_state=RANDOM_STATE),
        AdaBoostClassifier(random_state=RANDOM_STATE),
        MLPClassifier(max_iter=1000, random_state=RANDOM_STATE)
    ]

    param_tests = [
        {
            'max_depth': range(1, 11, 1)
        },
        {
            'n_estimators': range(10, 210, 10),
            'max_depth': range(1, 11, 1)
        },
        {
            'n_estimators': range(10, 310, 10),
            'learning_rate': [0.001, 0.01, 0.1, 1]
        },
        {
            'hidden_layer_sizes': [(10), (50), (100), (10, 10), (10, 10, 10)]
        }
    ]

    for i in range(0, 4, 1):
        gsearch = GridSearchCV(estimators[i] , param_grid = param_tests[i], scoring='roc_auc', cv=5 )
        gsearch.fit(X, y)
        joblib.dump(gsearch.best_estimator_, './../../out/best_model/' + names[i] + '.pkl')
        print("Best score: %0.3f" % gsearch.best_score_)
        print("Best parameters set:")
        best_parameters = gsearch.best_estimator_.get_params()
        for param_name in sorted(param_tests[i].keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))


def evaluate(X_train, X_test, y_train, y_test):
    names = ["MyDT", "MyRF", "MyAdaBoost", "MyMLP"]
    
    classifiers = []

    for i in range(0, 4, 1):
        classifiers.append(joblib.load('./../../out/best_model/' + names[i] + '.pkl'))
    
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