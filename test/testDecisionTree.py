#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 16:43:00 2017

@author: Mirza Elahi (me5vp)
"""
import sys
# appending the path to PubPyPlot
sys.path.append('../src/') 
sys.path.append('../data/')  

from decisionTree import decisionTree

def main():
    
    dT = decisionTree( enableLoggingTime=True )
    #load data
    dT.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 280080)
    #split train and test by 70-30 ratio
    dT.testTrainSplit( dT.feature, dT.Class, test_size = 0.3)
    #load model
    dT.loadModel()
    print( dT.toString() )
    #train model
    dT.trainModel( featureTrain = dT.fTrain, classTrain = dT.cTrain )
    #test model
    dT.classPred = dT.testModel( featureTest = dT.fTest )
    #metrices
    accuracy, matConf, matCohenKappa, \
    strClassificationReport = dT.getMetrics( classTest = dT.cTest, 
                                                 classPred = dT.classPred,
                                                 boolPrint = True)
    # cmap figure generation for confusion matrix
    dT.printConfusionMatrix( matConf )
    
    
if __name__ == '__main__':
    
    main()