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
    
    rf = randomForrest()
    rf.loadData( fileName = '../data/creditcard.csv', colClass=31)
    
    rf.testTrainSplit( rf.feature, rf.Class, test_size = 0.2)
    
    rf.trainModel( featureTrain = rf.fTrain, classTrain = rf.cTrain )
    
    rf.classPred = rf.testModel( featureTest = rf.fTest )
    
    accuracy, matConf, matCohenKappa, \
    strClassificationReport = rf.confusionMetric( classTest = rf.cTest, 
                                                 classPred = rf.classPred)
    
    print("Avg. Precision Score = %0.2f %%", accuracy*100)
    
    
if __name__ == '__main__':
    main()