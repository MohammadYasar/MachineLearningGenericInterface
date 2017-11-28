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

from adaBoost import adaBoost
from predictor import scoring

def main():
    
    adaB = adaBoost( enableLoggingTime=True )
    #load data
    adaB.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 284808)
    # feature scaling
    adaB.scaleFeature( minm=0, maxm=1 )
    #Feature reduction (loading previously saved data)
    feaSelecData = adaB.loadVariables( 'featureExtractAll' )
    adaB.selectImportantFeatures( feaSelecData['selectedIndices'] )
    # sweeping parameters lists
    adaB.n_estimatorsSweep = [31, 51, 71]

    # do double cross
    ValScoreList, \
    ValScoreStdList, \
    TestScoreList, \
    bestParamList, \
    allData = adaB.doubleCrossValidate(adaB.featureNumpy, 
                                             adaB.ClassNumpy,
                                             scoring=scoring.MCC,
                                             nFoldOuter=5, nFoldInner=4,
                                             isStratified = True,
                                             fileName='adaBoost/adaBoostData')
    
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