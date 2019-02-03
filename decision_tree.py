# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:08:33 2019

@author: ihuang9
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from helper_methods import loadUCIBreastCancerData, loadCalTechData
from helper_methods import performGridSearch
from mlxtend.plotting import plot_learning_curves
import graphviz
from sklearn import tree

#Load datasets
X_train, X_test, y_train, y_test = loadUCIBreastCancerData()
#X_train, X_test, y_train, y_test = loadCalTechData()


#running grid search on n_neighbor with distance function
model = DecisionTreeClassifier()
params = {"max_depth": np.arange(1, 20),
          "min_samples_split": np.arange(2, 7)}
param_name = ["max_depth", "min_samples_split"]
verbal_param_name = ["1. Max Depth", "2. Min Samples Split"]
performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)

model = DecisionTreeClassifier(max_depth = 6, min_samples_split = 6)
model.fit(X_train, y_train)
dot_data = tree.export_graphviz(model)
graph = graphviz.Source(dot_data)


#run the weight against n_neighbors on grid search
#model = DecisionTreeClassifier(metric = "manhattan")
#param_name = ["n_neighbors", "weights"] 
#params = {param_name[0]: np.arange(1, 31),
#          param_name[1]: ["uniform", "distance"]}
#verbal_param_name = ["N Neighbors", "Weights"]
#performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)


##running learning curve on the best hyperparameters
##caltech dataset
#model = KNeighborsClassifier(metric = "manhattan", n_neighbors = 1)
#plot_learning_curves(np.array(X_train), y_train, np.array(X_test), y_test, model, scoring = "accuracy")
##uci dataset
#model = KNeighborsClassifier(metric = "manhattan", n_neighbors = 3)
#plot_learning_curves(np.array(X_train), y_train, np.array(X_test), y_test, model, scoring = "f1_weighted")