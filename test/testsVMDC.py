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

def main():
    
    mySVM = sVM( enableLoggingTime=True )
    #load data
    mySVM.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 30808)
    
    feaSelecData = mySVM.loadVariables( 'featureExtractAll' )
    mySVM.selectImportantFeatures( feaSelecData['selectedIndices'] )
    
    mySVM.kernelSweep = ['linear', 'poly', 'rbf']
    mySVM.CSweep = [1, 10]
    mySVM.gammaSweep = ['auto', 5]
    # do double cross
    ValAccuList, \
    ValStdList, \
    TestAccuList, \
    bestParamList = mySVM.doubleCrossValidate(mySVM.featureNumpy, 
                                             mySVM.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4)
    print ValAccuList
    print ValStdList
    print TestAccuList
    print bestParamList
if __name__ == '__main__':
    
    main()