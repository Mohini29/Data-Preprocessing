# Data Preprocessing Class

#importing numpy - mathematical library
import numpy as np
# For ploting the data
import matplotlib.pyplot as plt
# import and manage Data sets
import pandas as pd


import os
os.chdir('C:/Users/I325765/Documents/Machine-Learning-AZ/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing\Data_Preprocessing')
#importing data 
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

#Replacing the missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])

#Encode the categorical variables in data using label encoder and dummy encoder using oneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)

#Seperating the training set and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, train_size=0.8, random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
SC_X  = StandardScaler()
X_train = SC_X.fit_transform(X_train)
X_test = SC_X.transform(X_test)