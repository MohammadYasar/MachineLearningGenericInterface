#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:17:21 2017

@author: Mirza Elahi (me5vp), Shabnam Wahed (sw2wm)
"""
import sys
# appending the path to PubPyPlot
sys.path.append('../src/') 
sys.path.append('../data/')  

from randomForrest import randomForrest

def main():
    
    rf = randomForrest( enableLoggingTime=True )
    #load data
    rf.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 280808)
    #split train and test by 80-20 ratio
    rf.testTrainSplit( rf.feature, rf.Class, test_size = 0.3)
    #train model
    rf.trainModel( featureTrain = rf.fTrain, classTrain = rf.cTrain )
    #test model
    rf.classPred = rf.testModel( featureTest = rf.fTest )
    #metrices
    accuracy, matConf, matCohenKappa, \
    strClassificationReport = rf.getMetrics( classTest = rf.cTest, 
                                                 classPred = rf.classPred)
    
    print('Avg. Precision Score = %0.2f %%\n' % (accuracy*100) )
    print('Classification Report:\n = %s\n' % (strClassificationReport) )
    print('Confusion Matrix:\n' )
    print(matConf)
    print('\n')
    
    rf.printConfusionMatrix( matConf )
    
    
if __name__ == '__main__':
    
    main()