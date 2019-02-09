# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:08:33 2019

@author: ihuang9
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from helper_methods import loadUCIBreastCancerData, loadCalTechData
from helper_methods import performGridSearch
from mlxtend.plotting import plot_learning_curves
import graphviz
from sklearn import tree

#Load datasets
#X_train, X_test, y_train, y_test = loadUCIBreastCancerData()
X_train, X_test, y_train, y_test = loadCalTechData()


#running grid search on n_neighbor with distance function
model = RandomForestClassifier(n_estimators = 100)
#params = {"max_depth": np.arange(1, 100),
#          "min_samples_split": np.arange(2, 7)}
params = {"max_depth": np.arange(1, 100, 3),
          "criterion": ["gini", "entropy"]}
param_name = ["max_depth", "criterion"]
verbal_param_name = ["Max Depth", "Criterion"]
performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)

model = RandomForestClassifier(max_depth = 6, min_samples_split = 6)
model.fit(X_train, y_train)
dot_data = tree.export_graphviz(model)
graph = graphviz.Source(dot_data)

#running learning curve on the best hyperparameters
model = RandomForestClassifier(criterion = "entropy", max_depth = 22)
plot_learning_curves(np.array(X_train), y_train, np.array(X_test), y_test, model, scoring = "accuracy")