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
from skimage.filters import sobel
from sklearn import datasets
from skimage.color import rgb2gray
from skimage.feature import hog
import matplotlib.pyplot as plt
import seaborn as sns

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
#                img = scipy.misc.imresize(imread(fullpath), [128,128, 3])
                img = resize(imread(fullpath), [128,128, 3], mode='reflect', anti_aliasing=True)
#                img = img.astype('float32')
#                img[:,:,0] -= 123.68
#                img[:,:,1] -= 116.78
#                img[:,:,2] -= 103.94
                img = rgb2gray(img)
                img = sobel(img)
                img = hog(img, orientations=8, pixels_per_cell=(32, 32), cells_per_block=(1, 1), block_norm="L2-Hys")
                imgs.append(img) # NORMALIZE IMAGE 
                label_curr = i
                labels.append(label_curr)
            #iter = (iter+1)%10;
    print ("Num imgs: %d" % (len(imgs)))
    print ("Num labels: %d" % (len(labels)) )
    print (ncategories)
    
    seed = 7
    np.random.seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(imgs, labels, test_size = test_size)
#        X_train = np.stack(X_train, axis=0)
#        y_train = np.stack(y_train, axis=0)
#        X_test = np.stack(X_test, axis=0)
#        y_test = np.stack(y_test, axis=0)
    print ("Num train_imgs: %d" % (len(X_train)))
    print ("Num test_imgs: %d" % (len(X_test)))
    # # one hot encode outputs
    enc = OneHotEncoder(categories='auto')
    y_train = enc.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
    y_test = enc.transform(np.array(y_test).reshape(-1, 1)).toarray()
    return X_train, X_test, y_train, y_test
#        num_classes= y_test.shape[1]
    
#        print(y_test.shape)
#        print(X_train[1,1,1,:])
#        print(y_train[1])
#        # normalize inputs from 0-255 to 0.0-1.0
#    print(X_train.shape)
#    print(X_test.shape)
#        X_train = X_train.transpose(0, 3, 1, 2)
#        X_test = X_test.transpose(0, 3, 1, 2)
#        print(X_train.shape)
#        print(X_test.shape)

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

def loadSuperconductorData():
    #dir_path = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip"
    dir_path = r"C:\Users\User\Downloads\Supervised_Learning\superconductor_data\train.csv"
    #dir_path = Path("E:\Google Drive\#Drive\Homework\CS4641\Supervised_Learning\superconductor_data\train.csv")
    
    dataset = pd.read_csv(dir_path)
    
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train = scaler.transform(X_train) 
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

