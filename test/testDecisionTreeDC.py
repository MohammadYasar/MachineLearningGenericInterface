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

def main():
    
    dT = decisionTree( enableLoggingTime=True )
    #load data
    dT.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 20080)
    feaSelecData = dT.loadVariables( 'featureExtractAll' )
    dT.selectImportantFeatures( feaSelecData['selectedIndices'] )
    
    dT.maxDepthSweep = [31, 51, None]
    
    ValAccuList, \
    ValStdList, \
    TestAccuList, \
    bestParamList, \
    allData = dT.doubleCrossValidate(dT.featureNumpy, 
                                             dT.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4,
                                             fileName='dTData')
    
    print ValAccuList
    print ValStdList
    print TestAccuList
    print bestParamList
if __name__ == '__main__':
    
    main()