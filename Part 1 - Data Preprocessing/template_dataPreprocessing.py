# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 10:15:49 2020

@author: Melody vilas pereira
"""
import pandas as pd
import numpy as np
import matplotlib.pyplotlib as plt

#importing dataset
dataset=pd.read_csv('Data.csv') #to read dataset
X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,3].values

#spliting dataset into training set and testset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.2, random_state = 0)

