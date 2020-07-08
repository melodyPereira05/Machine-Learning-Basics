# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:31:11 2020
@author: Melody vilas pereira
"""
#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd #to inport datasets
dataset=pd.read_csv('Data.csv') #to read dataset
X=dataset.iloc[:,:-1].values 
y=dataset.iloc[:,3].values
#taking care missing data
#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
#imputer = imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])

# Taking care of missing data
from sklearn.impute import SimpleImputer #to make ml models
#Imputer is a class
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',verbose=0) #object
imputer=imputer.fit(X[:,1:3]) #to fit to matrix X
X[:,1:3]=imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
#labelencoder_X=LabelEncoder()
#X[:,0]=labelencoder_X.fit_transform(X[:,0])
#onehotencoder=OneHotEncoder(ColumnTransformer=[0])
#X=onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#sc_y = StandardScaler()
#y_train = sc_y.fit_transform(y_train.reshape(-1,1))