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

def main():
    
    ckNN = kNN( enableLoggingTime=True )
    #load data
    ckNN.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 284808)
    #Feature reduction (loading previously saved data)
    feaSelecData = ckNN.loadVariables( 'featureExtractAll' )
    ckNN.selectImportantFeatures( feaSelecData['selectedIndices'] )
    
    ckNN.kSweep = [1, 3, 5]
    # do double cross
    ValAccuList, \
    ValStdList, \
    TestAccuList, \
    bestParamList, \
    allData = ckNN.doubleCrossValidate(ckNN.featureNumpy, 
                                             ckNN.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4,
                                             fileName='kNN/kNNData')
    print ValAccuList
    print ValStdList
    print TestAccuList
    print bestParamList
if __name__ == '__main__':
    
    main()