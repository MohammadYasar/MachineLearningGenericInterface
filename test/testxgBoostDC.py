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

from xgBoost import xgBoost
from predictor import scoring

def main():
    
    xgb = xgBoost( enableLoggingTime=True )
    #load data
    xgb.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 284808)
    # feature scaling
    xgb.scaleFeature( minm=0, maxm=1 )
    # Reducing feature dimension based on importance
    feaSelectData = xgb.loadVariables( 'featureExtractAll' )
    xgb.selectImportantFeatures( feaSelectData['selectedIndices'] )
    
    # sweeping parameters lists
    xgb.max_depthSweep = [6, 10, 14]
    xgb.n_estimatorsSweep = [51, 71, 91]
    
    
    ValScoreList, \
    ValScoreStdList, \
    TestScoreList, \
    bestParamList, \
    allData = xgb.doubleCrossValidate(xgb.featureNumpy, 
                                             xgb.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4,
                                             scoring=scoring.MCC,
                                             isStratified = True,
                                             fileName='xgBoost/xgBoostData')
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