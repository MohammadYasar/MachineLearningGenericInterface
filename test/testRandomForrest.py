#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  10 09:17:21 2017

@author: Mirza Elahi (me5vp)
"""
import sys
# appending the path to PubPyPlot
sys.path.append('../src/') 
sys.path.append('../data/')  

from randomForrest import randomForrest
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    rf = randomForrest( enableLoggingTime=True )
    #load data
    rf.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 284808)
    #split train and test by 80-20 ratio
    rf.testTrainSplit( rf.feature, rf.Class, test_size = 0.2)
    print rf.feature
    #load model
    rf.loadModel(n_estimators=75, oob_score=True, n_jobs=2)
    print( rf.toString() )
    #train model
    rf.trainModel( featureTrain = rf.fTrain, classTrain = rf.cTrain )
    # finding importances
    importances = rf.model.feature_importances_
    threshold = 0.02
    # plot
    rf.plotImportances(importances, threshold, fileName='featureImportance')
    
    #test model
    rf.classPred = rf.testModel( featureTest = rf.fTest )
    #metrices
    accuracy, avgPrecScore, matConf, matCohenKappa, \
    strClassificationReport = rf.getMetrics( classTest = rf.cTest, 
                                                 classPred = rf.classPred,
                                                 boolPrint = False)
    # cmap figure generation for confusion matrix
    #rf.printConfusionMatrix( matConf )
    #Saving Importances in file
    saveData = rf.featureSelectImportance( importances, threshold = threshold)
    rf.saveVariables( saveData, 'featureExtractAll' )
    # testing load importances from file
    printData = rf.loadVariables( 'featureExtractAll' )
    rf.selectImportantFeatures(printData['selectedIndices'])
    #print rf.feature
    
if __name__ == '__main__':
    
    main()