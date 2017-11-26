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
from predictor import scoring

def main():
    
    rf = randomForrest( enableLoggingTime=True )
    #load data
    rf.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 284808)
    # feature scaling
    rf.scaleFeature( minm=0, maxm=1 )
    #Feature reduction (loading previously saved data)
    feaSelecData = rf.loadVariables( 'featureExtractAll' )
    rf.selectImportantFeatures( feaSelecData['selectedIndices'] )
    
    rf.n_estimatorsSweep = [51, 71, 91]
    # do double cross
    ValScoreList, \
    ValScoreStdList, \
    TestScoreList, \
    bestParamList, \
    allData = rf.doubleCrossValidate(rf.featureNumpy, rf.ClassNumpy, 
                                           nFoldOuter=5, nFoldInner=4,
                                           scoring=scoring.MCC,
                                           isStratified = True,
                                           fileName='rF/rFData')
    print "Validation Avg. Score for outer folds with best param: \n"
    print ValScoreList
    print "Validation Score Std. for outer folds with best param: \n"
    print ValScoreStdList
    print "Test Avg. Score for outer folds with best param: \n"
    print TestScoreList
    print "Best Param list for outer folds: \n"
    print bestParamList
if __name__ == '__main__': 
    main()