# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 10:07:02 2019

@author: ihuang9
"""

import numpy as np
from sklearn.svm import SVC, LinearSVC
from helper_methods import loadUCIBreastCancerData, loadCalTechData
from helper_methods import performGridSearch
from mlxtend.plotting import plot_learning_curves

#Load datasets
#X_train, X_test, y_train, y_test = loadUCIBreastCancerData()
X_train, X_test, y_train, y_test = loadCalTechData()


#running grid search on n_neighbor with distance function
model = SVC(gamma = "scale")
params = {"C": [10**x for x in range(-3, 10)],
          "kernel": ["linear", "poly", "rbf", "sigmoid"]}
param_name = ["C", "kernel"]
verbal_param_name = ["C", "Kernel"]
#performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)

#run the weight against n_neighbors on grid search
model = LinearSVC()
params = {"C": [10**x for x in range(-3, 10)],
          "multi_class": ["ovr", "crammer_singer"]}
param_name = ["C", "multi_class"]
verbal_param_name = ["C", "multi_class"]
#performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)


#running learning curve on the best hyperparameters
model = SVC(gamma = "scale", C = 10**3, kernel = "poly")
plot_learning_curves(np.array(X_train), y_train, np.array(X_test), y_test, model, scoring = "accuracy")