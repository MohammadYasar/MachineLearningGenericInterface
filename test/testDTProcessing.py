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

import numpy as np

def main():
    
    dT = decisionTree( enableLoggingTime=True )
    
    data = dT.loadVariables(fileName='dTData')
    ValAccuList = data['ValAccuList'] 
    ValStdList = data['ValStdList']
    TestAccuList = data['TestAccuList']
    bestParamList = data['bestParamList']
    # Fold data [OuterFoldNo][ParamListIndex][Accu/Conf][InnerFoldNo]
    OuterInnerFoldData = data['OuterInnerFoldData'] 
    sweepingList = data['sweepingList']
    OuterFoldNo = data['OuterFoldNo']
    InnerFoldNo = data['InnerFoldNo']
    bestParamIndexInSwpList = data['bestParamIndexInSwpList']

    
    print( "Fold no - Outer %d, Inner %d\n" % (OuterFoldNo, InnerFoldNo) )
    print( "Sweeping param list: %s\n" % (sweepingList) )
    #print( "Each Inner Fold Size: %d\n" % len(eachInnerFold[0][1]) )
    print( "best Params: %s\n" % (bestParamList)  )
    print(  "best Params List index in paramList : %s\n" % (bestParamIndexInSwpList)  )
    print( "Outer Fold Size: %d\n" % len(OuterInnerFoldData) )
    
    
    # Fold 1
    for outI in range( OuterFoldNo ):
        print( "Fold #%d\n" % (outI) )
        FoldData = OuterInnerFoldData[outI]
        bestParamAccu = np.array( FoldData[bestParamIndexInSwpList[outI]][0] )
        bestParamConf = np.array( FoldData[bestParamIndexInSwpList[outI]][1] )
        print( "\tBest param Avg. Valid. Accu. = %0.8f\n"  % ( np.mean(bestParamAccu) ) )
        for inn in range( InnerFoldNo ):
            fileName = "dT/O%d_I%d_conf" % (outI, inn)
            dT.printConfusionMatrix( bestParamConf[inn], filename=fileName )
if __name__ == '__main__':
    
    main()