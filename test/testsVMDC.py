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

from sVM import sVM
from predictor import scoring

def main():
    
    mySVM = sVM( enableLoggingTime=True )
    #load data
    mySVM.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 28480)
    # feature scaling
    mySVM.scaleFeature( minm=0, maxm=1 )
    #Feature reduction (loading previously saved data)
    feaSelecData = mySVM.loadVariables( 'featureExtractAll' )
    mySVM.selectImportantFeatures( feaSelecData['selectedIndices'] )
    # setting parameters
    mySVM.max_iter = 1E5
    # sweeping parameter lists
    mySVM.kernelSweep = ['linear', 'poly']
    mySVM.CSweep = [1]
    mySVM.gammaSweep = ['auto', 1]
    
    # do double cross
    ValScoreList, \
    ValScoreStdList, \
    TestScoreList, \
    bestParamList, \
    allData = mySVM.doubleCrossValidate(mySVM.featureNumpy, 
                                             mySVM.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4,
                                             scoring=scoring.MCC,
                                             isStratified = True,
                                             fileName='sVM/sVMData')
    
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