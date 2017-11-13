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
    mySVM.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 284808)
    #Feature reduction (loading previously saved data)
    feaSelecData = mySVM.loadVariables( 'featureExtractAll' )
    mySVM.selectImportantFeatures( feaSelecData['selectedIndices'] )
    mySVM.max_iter = 100000
    mySVM.kernelSweep = ['linear', 'poly']
    mySVM.CSweep = [1]
    mySVM.gammaSweep = ['auto', 1]
    # do double cross
    ValAccuList, \
    ValStdList, \
    TestAccuList, \
    bestParamList, \
    allData = mySVM.doubleCrossValidate(mySVM.featureNumpy, 
                                             mySVM.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4,
                                             fileName='sVM/sVMData')
    print ValAccuList
    print ValStdList
    print TestAccuList
    print bestParamList
if __name__ == '__main__':
    
    main()