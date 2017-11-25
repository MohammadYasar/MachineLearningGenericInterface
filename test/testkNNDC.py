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
from predictor import scoring

def main():
    
    ckNN = kNN( enableLoggingTime=True )
    #load data
    ckNN.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 28480)
    # feature scaling
    ckNN.scaleFeature( minm=0, maxm=1 )
    #Feature reduction (loading previously saved data)
    feaSelecData = ckNN.loadVariables( 'featureExtractAll' )
    ckNN.selectImportantFeatures( feaSelecData['selectedIndices'] )
    
    ckNN.kSweep = [1, 3, 5]
    # do double cross
    ValScoreList, \
    ValScoreStdList, \
    TestScoreList, \
    bestParamList, \
    allData = ckNN.doubleCrossValidate(ckNN.featureNumpy, 
                                             ckNN.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4,
                                             scoring=scoring.MCC,
                                             isStratified = True,
                                             fileName='kNN/kNNData')
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