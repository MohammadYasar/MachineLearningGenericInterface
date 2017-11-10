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

from kNN import kNN

def main():
    
    ckNN = kNN( enableLoggingTime=True )
    #load data
    ckNN.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 28008)
    #split train and test by 80-20 ratio
    ckNN.testTrainSplit( ckNN.feature, ckNN.Class, test_size = 0.3)
    #load model
    ckNN.loadModel(n_neighbors=3)
    print( ckNN.toString() )
    #train model
    ckNN.trainModel( featureTrain = ckNN.fTrain, classTrain = ckNN.cTrain )
    #test model
    ckNN.classPred = ckNN.testModel( featureTest = ckNN.fTest )
    #metrices
    accuracy, matConf, matCohenKappa, \
    strClassificationReport = ckNN.getMetrics( classTest = ckNN.cTest, 
                                                 classPred = ckNN.classPred,
                                                 boolPrint = True)
    # cmap figure generation for confusion matrix
    ckNN.printConfusionMatrix( matConf )
    
    
if __name__ == '__main__':
    
    main()