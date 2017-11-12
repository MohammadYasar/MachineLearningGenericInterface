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

def main():
    
    adaB = adaBoost( enableLoggingTime=True )
    #load data
    adaB.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 28480)
    #Feature reduction (loading previously saved data)
    feaSelecData = adaB.loadVariables( 'featureExtractAll' )
    adaB.selectImportantFeatures( feaSelecData['selectedIndices'] )
    
    adaB.n_estimatorsSweep = [5, 9, 13]

    # do double cross
    ValAccuList, \
    ValStdList, \
    TestAccuList, \
    bestParamList, \
    allData  = adaB.doubleCrossValidate(adaB.featureNumpy, 
                                             adaB.ClassNumpy,
                                             nFoldOuter=5, nFoldInner=4,
                                             fileName='adaBoost/adaBoostData')
    print ValAccuList
    print ValStdList
    print TestAccuList
    print bestParamList
if __name__ == '__main__':
    
    main()