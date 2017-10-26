# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 11:49:32 2017

@author: shabnam
"""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
#from sklearn.metrics import matthews_corrcoef

#Read dataset from file and visualize them
data=pd.read_csv('file:///C:/Users/shabnam/Desktop/project_ML/creditcard.csv')
print(data)
print(data.describe())
plt.figure(figsize=(16,8))
for w in range(1, 29):
    plt.plot(data['V'+str(w)])
plt.legend(loc='best', fontsize=16, ncol=8)
plt.xlabel("Plot of Features")

f, (axis1, axis2) = plt.subplots(2, 1, sharex=True, figsize=(10,6))
axis1.hist(data.Amount[data.Class == 1], bins = 30)
axis1.set_title('Fradulent Transaction')
axis2.hist(data.Amount[data.Class == 0], bins = 30)
axis2.set_title('Normal Transaction')

X = data.iloc[:,1:29]
Y = data.Class

# normalization is done to ensure the distribution of each feature same
#def normalization(X):
    #for features in X.columns:
        #X[features] -= X[features].mean()
        #X[features] /= X[features].std()
    #return X

# training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state=0)

# Synthetic Minority Over-sampling TEchnique SMOTE is used to create a new balanced data set from training dataset as our dataset is overly imbalanced
oversampler=SMOTE(random_state=0)
os_X,os_Y=oversampler.fit_sample(X_train,Y_train)
len(os_Y[os_Y==1])

# performing logistic regression
lr = LogisticRegression()
lr.fit(os_X, os_Y)
# perform prediction on testing data
Y_pred = lr.predict(X_test)
c_matrix = confusion_matrix(Y_test, Y_pred)
print "Confusion matrix for logistic regression= \n", c_matrix
print "Classification report for logistic regression= \n", classification_report(Y_test, Y_pred)
#print " MCC score for logistic regression= " + str(np.round(matthews_corrcoef(Y_test, Y_pred)))
print "Cohen Kappa for logistic regression= " + str(np.round(cohen_kappa_score(Y_test, Y_pred),3))
print "Accuracy for logistic regression=" + str(np.round(100*float((c_matrix[0][0]+c_matrix[1][1]))/float((c_matrix[0][0]+c_matrix[1][1] + c_matrix[1][0] + c_matrix[0][1])),2))+'%'

# Performing Random Forest 
rf = RandomForestClassifier(n_estimators=100, oob_score=True, n_jobs=4)
rf.fit(os_X, os_Y)
# perform predictions on testing data
Y_pred = rf.predict(X_test)
c_matrix1 = confusion_matrix(Y_test, Y_pred)
print "Confusion matrix for ramdom forest= \n", c_matrix1
print "Classification report for random forest= \n", classification_report(Y_test, Y_pred)
#print " MCC score for random forest= " + str(np.round(matthews_corrcoef(Y_test, Y_pred)))
print "Cohen Kappa for random forest=" + str(np.round(cohen_kappa_score(Y_test, Y_pred),3))
print "Accuracy for random forest=" + str(np.round(100*float((c_matrix1[0][0]+c_matrix1[1][1]))/float((c_matrix1[0][0]+c_matrix1[1][1] + c_matrix1[1][0] + c_matrix1[0][1])),2))+'%'
#print "Recall:" + str(np.round(100*float((c_matrix1[1][1]))/float((c_matrix1[1][0]+c_matrix1[1][1])),2))+'%'
# less value of Recall than Logistic Regression