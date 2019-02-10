# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 17:40:03 2019

@author: ihuang9
"""

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from helper_methods import loadUCIBreastCancerData, loadCalTechData
from helper_methods import performGridSearch
from mlxtend.plotting import plot_learning_curves
import graphviz
from sklearn.tree import DecisionTreeClassifier

#Load datasets
#X_train, X_test, y_train, y_test = loadUCIBreastCancerData()
X_train, X_test, y_train, y_test = loadCalTechData()
X_train = np.array(X_train)
y_train = np.array(y_train)

#running grid search on n_neighbor with distance function
estimator = DecisionTreeClassifier()
model = AdaBoostClassifier(base_estimator = estimator)
#params = {"max_depth": np.arange(1, 100),
#          "min_samples_split": np.arange(2, 7)}
params = {"n_estimators": np.arange(1, 100, 10),
          "algorithm": ["SAMME", "SAMME.R"]}
param_name = ["n_estimators", "algorithm"]
verbal_param_name = ["n_estimators", "algorithm"]
#performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)

#running grid search on n_neighbor with distance function
model = GradientBoostingClassifier()
#params = {"max_depth": np.arange(1, 100),
#          "min_samples_split": np.arange(2, 7)}
params = {"max_depth": np.arange(1, 100, 5),
          "loss": ["deviance", "exponential"]}
param_name = ["max_depth", "loss"]
verbal_param_name = ["max_depth", "loss"]
#performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)

model = GradientBoostingClassifier()
#params = {"max_depth": np.arange(1, 100),
#          "min_samples_split": np.arange(2, 7)}
params = {"n_estimators": np.arange(1, 100, 5),
          "loss": ["deviance"]}
param_name = ["n_estimators", "loss"]
verbal_param_name = ["n_estimators", "loss"]
performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)


#running learning curve on the best hyperparameters
#model = AdaBoostClassifier(criterion = "entropy", max_depth = 22)
#plot_learning_curves(np.array(X_train), y_train, np.array(X_test), y_test, model, scoring = "accuracy")