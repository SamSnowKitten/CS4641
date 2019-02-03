# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:53:28 2019

@author: I-Ping Huang
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from helper_methods import loadUCIBreastCancerData, loadCalTechData
from helper_methods import performGridSearch
from mlxtend.plotting import plot_learning_curves

#Load datasets
#X_train, X_test, y_train, y_test = loadUCIBreastCancerData()
X_train, X_test, y_train, y_test = loadCalTechData()


#running grid search on n_neighbor with distance function
model = KNeighborsClassifier()
params = {"n_neighbors": np.arange(1, 31),
          "metric": ["chebyshev", "minkowski", "manhattan"]}
param_name = ["n_neighbors", "metric"]
verbal_param_name = ["N Neighbors", "Distance Function"]
performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)

#run the weight against n_neighbors on grid search
model = KNeighborsClassifier(metric = "manhattan")
param_name = ["n_neighbors", "weights"] 
params = {param_name[0]: np.arange(1, 31),
          param_name[1]: ["uniform", "distance"]}
verbal_param_name = ["N Neighbors", "Weights"]
performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)


#running learning curve on the best hyperparameters
#caltech dataset
model = KNeighborsClassifier(metric = "manhattan", n_neighbors = 1)
plot_learning_curves(np.array(X_train), y_train, np.array(X_test), y_test, model, scoring = "accuracy")
#uci dataset
model = KNeighborsClassifier(metric = "manhattan", n_neighbors = 3)
plot_learning_curves(np.array(X_train), y_train, np.array(X_test), y_test, model, scoring = "f1_weighted")