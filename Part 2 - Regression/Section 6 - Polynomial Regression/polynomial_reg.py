# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 09:55:39 2020

@author: Melody vilas pereira
"""
#importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#data preprocessing

dataset=pd.read_csv('Position_Salaries.csv')

X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#from sklearn.model_selection import test_train_split
#X_train,X_test,y_train,y_test=test_train_split(X,y,test_size=,random_state=0)

#Fitting linear Regression to the dataset

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression() 
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=3)
X_poly=poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)

X_grid = np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg.predict(X_grid),color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()


plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Truth or bluff')
plt.xlabel('Position')
plt.ylabel('Salaries')
plt.show()

lin_reg.predict(6.5)

lin_reg2.predict(poly_reg.fit_transform(6.5))

