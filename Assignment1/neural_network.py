# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 13:34:17 2019

@author: ihuang9
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from helper_methods import loadUCIBreastCancerData, loadCalTechData
from helper_methods import performGridSearch
from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt

#Load datasets
#X_train, X_test, y_train, y_test = loadUCIBreastCancerData()
X_train, X_test, y_train, y_test = loadCalTechData()


#running grid search on n_neighbor with distance function
model = MLPClassifier()
params = {"hidden_layer_sizes": np.arange(1, 500, 50),
          "solver": ["lbfgs", "adam"]}
param_name = ["hidden_layer_sizes", "solver"]
verbal_param_name = ["Hidden Layer Sizes", "Solver"]
X_train = np.array(X_train)
y_train = np.array(y_train)
performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test)

def plotIterativeScore(mlp):
    N_EPOCHS = 500
    N_CLASSES = np.unique(y_train)
    
    scores_train = []
    scores_test = []
    
    # EPOCH
    epoch = 0
    while epoch < N_EPOCHS:
        print('epoch: ', epoch)
        mlp.partial_fit(X_train, y_train, classes=N_CLASSES)
        scores_train.append(mlp.score(X_train, y_train))
        scores_test.append(mlp.score(X_test, y_test))
        epoch += 1
        
    plt.plot(scores_train, color='green', alpha=0.8, label='Train')
    plt.plot(scores_test, color='magenta', alpha=0.8, label='Test')
    plt.title("Accuracy over epochs", fontsize=14)
    plt.xlabel('Epochs')
    plt.legend(loc='upper left')
    plt.show()
model = MLPClassifier()
plotIterativeScore(model)

##running learning curve on the best hyperparameters
model = MLPClassifier(hidden_layer_sizes = 451, solver = 'adam')
plot_learning_curves(X_train, y_train, X_test, y_test, model, scoring = "accuracy")