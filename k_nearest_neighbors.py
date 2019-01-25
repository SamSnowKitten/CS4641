# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:53:28 2019

@author: I-Ping Huang
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from helper_methods import loadUCIBreastCancerData, loadCalTechData
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import time
import matplotlib.pyplot as plt


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

#    plt.figure(1)
    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
#        plt.subplot(2, 2, 1 + idx)
#        print(idx, grid_param_1, scores_mean[idx,:], name_param_2 + ': ' + str(val))
        plt.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

#    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_title("Grid Search Scores")
#    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_xlabel(name_param_1)
#    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.set_ylabel('CV Average Score')
    ax.legend(loc="best")
#    ax.legend(loc="best", fontsize=15)
    ax.grid(True)

def performGridSearch(model, params, X_train, X_test, y_train, y_test):
    # construct the set of hyperparameters to tune
#    params = {"n_neighbors": np.arange(1, 31, 2),
#    	"metric": ["euclidean", "cityblock", "minkowski"]}
#    
#    model = KNeighborsRegressor()
    
    # tune the hyperparameters via a randomized search
    grid = GridSearchCV(model, params, cv=10)
    start = time.time()
    grid.fit(X_train, y_train)
         
    # evaluate the best randomized searched model on the testing
    # data
    print("[INFO] grid search took {:.2f} seconds".format(
    	time.time() - start))
    acc = grid.score(X_test, y_test)
    print("[INFO] grid search accuracy on test set: {:.2f}%".format(acc * 100))
    print("[INFO] grid search best parameters: {}".format(
    	grid.best_params_))
    plot_grid_search(grid.cv_results_, params["n_neighbors"], params["metric"], 'N Neighbors', 'Distance Function')


model = KNeighborsClassifier()
params = {"n_neighbors": np.arange(1, 31),
          "metric": ["chebyshev", "minkowski", "manhattan"]}
#X_train, X_test, y_train, y_test = loadUCIBreastCancerData()
X_train, X_test, y_train, y_test = loadCalTechData()
performGridSearch(model, params, X_train, X_test, y_train, y_test)
#regressor = KNeighborsClassifier(metric='cityblock', n_neighbors=3)
#regressor.fit(trainData, trainLabels)

#print(regressor.score(testData, testLabels))

#y_pred = regressor.predict(testData)

#error = []
#
## Calculating error for K values between 1 and 40
#for i in range(1, 40):
#    knn = KNeighborsClassifier(n_neighbors=i)
#    knn.fit(X_train, y_train)
#    error.append(knn.score(X_test, y_test))
#    
#plt.figure(figsize=(12, 6))
#plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',  
#         markerfacecolor='blue', markersize=10)
#plt.title('Error Rate K Value')
#plt.xlabel('K Value')
#plt.ylabel('Mean Error')