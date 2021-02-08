# coding=utf-8


# IFSIC : tocheck - link with sklearn. Type in terminal :
# export PYTHONPATH="/usr/local/anaconda2/lib/python2.7/site-packages/"

import os, sys
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score



################################
# LOAD DATASET
################################


base_dir = "/share/esir2/aci/img_data/cats_and_dogs_sampled/"
output_dir="./"

X_train = np.load(output_dir + "vgg16_train_descriptors.npy")
y_train = np.load(output_dir + "vgg16_train_target.npy")

X_test = np.load(output_dir + "vgg16_test_descriptors.npy")
y_test = np.load(output_dir + "vgg16_test_target.npy")
target_names = ['cat','dog']


print("Class distribution in train set: \n",np.unique(y_train, return_counts=True))
print("Class distribution in test set: \n", np.unique(y_test, return_counts=True))


"""
###### Features scaling
print("Before preprocessing: \n", X_train[1,1:10])
print("Before preprocessing: \n", X_test[1,1:10])
scaler = preprocessing.StandardScaler().fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
print("After standardization: \n", X_train[1,1:10])
print("After preprocessing: \n", X_test[1,1:10])
"""

## QUESTION : Que fait la fonction 'StandardScaler' ? Comment est-elle appliquée ?


################################
## LEARNING REGLOG CLASSIFIER
################################


print("# Learning a Logistic regression (default parameters) ")
clf_reglog = LogisticRegression()
clf_reglog.fit(X_train,y_train) # training



print("# Evaluation on the test set")
y_pred_reglog = clf_reglog.predict(X_test)
print("\n Accuracy : ")
print("Reglog (test set) : \t" + str(metrics.accuracy_score(y_test, y_pred_reglog)) + "\n")
print(metrics.classification_report(y_test, y_pred_reglog, target_names=target_names)) # support = The number of occurrences of each label in y_test
print("Confusion Matrix : \n", metrics.confusion_matrix(y_test, y_pred_reglog))


## QUESTION 1 : Quelle est le taux de bonnes classifications sur l'ensemble de test ?


################################
## TUNING HYPERPARAMETERS
################################

print("# Hyperparameters tuning")
param_grid = [
			        {'penalty': ['none'], 'solver':['newton-cg']},
              {'C': [0.001,0.01,1,5,10,25], 'penalty': ['l2'], 'solver':['newton-cg']},
              {'C': [0.001,0.01,1,5,10,25], 'penalty': ['l1'], 'solver':['saga']},
              {'C': [0.001,0.01,1,5,10,25], 'penalty': ['elasticnet'], 'solver':['saga'],'l1_ratio':[0.5]},
              ]

print("\n # Tuning hyper-parameters")
#clf = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, verbose=2)
clf = GridSearchCV(LogisticRegression(max_iter=500), param_grid, cv=5, verbose=2)
clf.fit(X_train, y_train)


# QUESTION 2 : Quels sont les hyperparametres evalues ? Combien de combinaison sont évaluées ?
# QUESTION 3 : Comment sont sélectionnés les meilleurs hyperparametres ?


print("Best parameters set found on training set:")
print(clf.best_params_)
print("Grid scores on developpment set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

# QUESTION 4 : A quoi correspondent les scores affichés ?


################################
## EVALUATION (after hyperparameters tuning)
################################

y_pred = clf.predict(X_test)
print("\n Reglog on test set (parameter tuning) : \t" + str(metrics.accuracy_score(y_test, y_pred)) + "\n")


sys.exit(0)


################################
## EVALUATION by cross validation
################################

# TODO : Reprenez les meilleurs hyperparametres trouves precedemment et evaluer le classifieur
# par validation croisee à 10 plis.
# Utilisez : cross_val_score(clf, X, y,cv=10)

# QUESTION 5 : Sur quel ensemble applique-t-on cette methode (evaluation par validation croisee) ?

# QUESTION 6 : Quel est le score obtenu ? A quoi correspond-il ?


sys.exit(0)



################################
## LEARNING SVM CLASSIFIER
################################

print("# Learning an SVM")
clf_svm = svm.SVC(gamma='auto') # default parameters
clf_svm.fit(X_train, y_train)


################################
## PREDICTION AND EVALUATION
################################

y_pred_svm = clf_svm.predict(X_test)

print("\n Accuracy : ")
print("SVM (test set) : \t" + str(metrics.accuracy_score(y_test, y_pred_svm)) + "\n")
print("Confusion Matrix : \n", metrics.confusion_matrix(y_test, y_pred_svm))


################################
## TUNING HYPERPARAMETERS
################################

# Set the (hyper)parameters by cross-validation
param_grid = [
              {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'degree': [2, 3, 4], 'kernel': ['poly']},
              {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.1, 1.0], 'kernel': ['rbf']},
              ]
print("\n # Tuning hyper-parameters")
clf = GridSearchCV(svm.SVC(), param_grid, cv=5, verbose=2)
clf.fit(X_train, y_train)


print("Best parameters set found on training set:")
print(clf.best_params_)
print("Grid scores on developpment set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


y_pred = clf.predict(X_test)
print("\n SVM on test set (parameter tuning) : \t" + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
print("Confusion Matrix : \n", metrics.confusion_matrix(y_test, y_pred_svm))


# QUESTION 7 : Quels sont les meilleurs hyperparamètres ?

# QUESTION 8 : Quels sont les scores obtenus avant/après tuning des hyperparamètres ?