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
from predictor import scoring

def main():
    
    dT = decisionTree( enableLoggingTime=True )
    #load data
    dT.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 284808)
    # feature scaling
    dT.scaleFeature( minm=0, maxm=1 )
    # Reducing feature dimension based on importance
    feaSelectData = dT.loadVariables( 'featureExtractAll' )
    dT.selectImportantFeatures( feaSelectData['selectedIndices'] )
    
    # sweeping parameters lists
    dT.maxDepthSweep = [31, 51, None]
    
    ValScoreList, \
    ValScoreStdList, \
    TestScoreList, \
    bestParamList, \
    allData = dT.doubleCrossValidate(dT.featureNumpy, 
                                             dT.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4,
                                             scoring=scoring.MCC,
                                             isStratified = True,
                                             fileName='dT/dTData')
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