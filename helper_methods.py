# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 15:13:57 2019

@author: I-Ping Huang
"""
import os, imageio
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Normalizer
from skimage.transform import resize
from sklearn.model_selection import GridSearchCV
import time
from skimage.filters import scharr
from sklearn import datasets
from skimage.color import rgb2gray
from skimage.feature import hog
import matplotlib.pyplot as plt

def loadCalTechData(test_size = 0.2):
    def imread(path):
        img = imageio.imread(path).astype(np.float)
        if len(img.shape) == 2:
            img = np.transpose(np.array([img, img, img]), (2, 0, 1))
        return img
        
    cwd = os.getcwd()
    path = cwd + "/101_ObjectCategories"
    valid_exts = [".jpg", ".gif", ".png", ".jpeg"]
    print ("[%d] CATEGORIES ARE IN \n %s" % (len(os.listdir(path)), path))
    
    categories = sorted(os.listdir(path))
    ncategories = len(categories)
    imgs = []
    labels = []
    # LOAD ALL IMAGES 
    for i, category in enumerate(categories):
        iter = 0
        for f in os.listdir(path + "/" + category):
            if iter == 0:
                ext = os.path.splitext(f)[1]
                if ext.lower() not in valid_exts:
                    continue
                fullpath = os.path.join(path + "/" + category, f)
                img = resize(imread(fullpath), [128,128, 3], mode='reflect', anti_aliasing=True)
                img = rgb2gray(img)
                img = scharr(img)
                img = hog(img, pixels_per_cell=(32, 32), block_norm="L2-Hys", transform_sqrt=True)
                imgs.append(img) # NORMALIZE IMAGE 
                label_curr = i
                labels.append(label_curr)
    print ("Num imgs: %d" % (len(imgs)))
    print ("Num labels: %d" % (len(labels)) )
    print (ncategories)
    
    seed = 7
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size = test_size)
    print ("Num train_imgs: %d" % (len(X_train)))
    print ("Num test_imgs: %d" % (len(X_test)))
    # # one hot encode outputs
    '''IMPORTANT, DON'T DO THIS IF THE CLASSIFIER DOES NOT TAKE THIS INPUT'''
    enc = OneHotEncoder(categories='auto')
    y_train = enc.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
    y_test = enc.transform(np.array(y_test).reshape(-1, 1)).toarray()
    return X_train, X_test, y_train, y_test

def loadUCIBreastCancerData(test_size = 0.2):
    bc = datasets.load_breast_cancer()
    bcdf = pd.DataFrame(bc.data, columns= bc.feature_names)
    bcdf['Diagnosis'] = bc.target
    X_train, X_test, y_train, y_test = train_test_split(bcdf.iloc[:,:-1], bcdf['Diagnosis'], test_size = test_size, random_state = 3)
    # Instantiate 
    norm = Normalizer()
    
    # Fit
    norm.fit(X_train)
    
    # Transform both training and testing sets
    X_train_norm = norm.transform(X_train)
    X_test_norm = norm.transform(X_test)
    return X_train_norm, X_test_norm, y_train, y_test

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    if (name_param_1[0] < name_param_2[0]):
        scores_mean = np.array(scores_mean).reshape(len(grid_param_1),len(grid_param_2))
        scores_mean = np.transpose(scores_mean)
    else:
        scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        plt.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
    ax.set_title("Grid Search Scores")
    ax.set_xlabel(name_param_1)
    ''''IMPORTANT, PLEASE ENABLE THIS FOR LOGARITHMIC PERFORMANCE'''
#    plt.xscale("log")
    ax.set_ylabel('CV Average Score')
    ax.legend(loc="best")
    ax.grid(True)

def performGridSearch(model, params, param_name, verbal_param_name, X_train, X_test, y_train, y_test):
    # tune the hyperparameters via a randomized search
    grid = GridSearchCV(model, params, cv=10)
    start = time.time()
    grid.fit(X_train, y_train)
         
    # evaluate the best randomized searched model on the testing data
    print("[INFO] grid search took {:.2f} seconds".format(
    	time.time() - start))
    acc = grid.score(X_test, y_test)
    print("[INFO] grid search accuracy on test set: {:.2f}%".format(acc * 100))
    print("[INFO] grid search best parameters: {}".format(
    	grid.best_params_))
    print("[INFO] grid search best average validation score: {:.2f}%".format(
    	grid.best_score_ * 100))
    plot_grid_search(grid.cv_results_, params[param_name[0]], params[param_name[1]], verbal_param_name[0], verbal_param_name[1])