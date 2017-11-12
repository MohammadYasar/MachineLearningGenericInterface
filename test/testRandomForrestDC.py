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

def main():
    
    rf = randomForrest( enableLoggingTime=True )
    #load data
    rf.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 10000)
    feaSelecData = rf.loadVariables( 'featureExtractAll' )
    rf.selectImportantFeatures( feaSelecData['selectedIndices'] )
    
    rf.n_estimatorsSweep = [31, 51, 71]
    # do double cross
    ValAccuList, \
    ValStdList, \
    TestAccuList, \
    bestParamList = rf.doubleCrossValidate(rf.featureNumpy, rf.ClassNumpy, 
                                           nFoldOuter=5, nFoldInner=4)
    print ValAccuList
    print ValStdList
    print TestAccuList
    print bestParamList
if __name__ == '__main__': 
    main()