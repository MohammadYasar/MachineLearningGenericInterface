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

def main():
    
#    rf = randomForrest( enableLoggingTime=True )
#    #load data
#    rf.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 28008)
#    #split train and test by 80-20 ratio
#    rf.testTrainSplit( rf.feature, rf.Class, test_size = 0.3)
#    #load model
#    rf.loadModel(n_estimators=50, oob_score=True, n_jobs=4)
#    print( rf.toString() )
#    #train model
#    rf.trainModel( featureTrain = rf.fTrain, classTrain = rf.cTrain )
#    #test model
#    rf.classPred = rf.testModel( featureTest = rf.fTest )
#    #metrices
#    accuracy, matConf, matCohenKappa, \
#    strClassificationReport = rf.getMetrics( classTest = rf.cTest, 
#                                                 classPred = rf.classPred,
#                                                 boolPrint = True)
#    # cmap figure generation for confusion matrix
#    rf.printConfusionMatrix( matConf )
    rf = randomForrest( enableLoggingTime=True )
    #load data
    rf.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 50008)
    rf.dataConvertToNumpy()
#    #split train and test by 80-20 ratio
#    ckNN.testTrainSplit( ckNN.feature, ckNN.Class, test_size = 0.3)
#    #load model
#    ckNN.loadModel(n_neighbors=3)
#    print( ckNN.toString() )
#    #train model
#    ckNN.trainModel( featureTrain = ckNN.fTrain, classTrain = ckNN.cTrain )
#    #test model
#    ckNN.classPred = ckNN.testModel( featureTest = ckNN.fTest )
#    #metrices
#    accuracy, matConf, matCohenKappa, \
#    strClassificationReport = ckNN.getMetrics( classTest = ckNN.cTest, 
#                                                 classPred = ckNN.classPred,
#                                                 boolPrint = True)
#    # cmap figure generation for confusion matrix
#    ckNN.printConfusionMatrix( matConf )
#    
    rf.doubleCrossValidate(rf.featureNumpy, rf.ClassNumpy)
    
if __name__ == '__main__':
    
    main()