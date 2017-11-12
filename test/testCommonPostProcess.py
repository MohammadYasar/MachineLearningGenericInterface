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

from predictor import predictor
import matplotlib.pyplot as plt
import numpy as np


def main():
    
    #processList = ['dT', 'kNN', 'rF']
    processList = ['adaBoost']
    count = 0
    for algo in processList:
    
        dT = predictor( enableLoggingTime=True )
        fileName = "%s/%sData" % (algo, algo)
        data = dT.loadVariables(fileName=fileName)
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
        
        AccuracyBoxPlot = np.ndarray(shape=(InnerFoldNo, OuterFoldNo), dtype=float )
        # Fold 1
        for outI in range( OuterFoldNo ):
            print( "Fold #%d\n" % (outI) )
            FoldData = OuterInnerFoldData[outI]
            bestParamAccu = np.array( FoldData[bestParamIndexInSwpList[outI]][0] )
            AccuracyBoxPlot[:, outI] = bestParamAccu
            bestParamConf = np.array( FoldData[bestParamIndexInSwpList[outI]][1] )
            print( "\tBest param Avg. Valid. Accu. = %0.8f\n"  % ( np.mean(bestParamAccu) ) )
            print( "\tBest param Valid. Accu. = \t\t")
            print bestParamAccu
            for inn in range( InnerFoldNo ):
                fileName = "%s/O%d_I%d_conf" % (algo, outI, inn)
                #dT.printConfusionMatrix( bestParamConf[inn], filename=fileName )
                
        print AccuracyBoxPlot
        # Create a figure instance
        fig = plt.figure(count, figsize=(8, 4))
    
        # Create an axes instance
        ax = fig.add_subplot(111)
        boxprops = dict(linewidth=3, color='#3852a3')
        medianprops = dict(linestyle='-', linewidth=2.5, color='#ed1e23')
        # Create the boxplot
        plt.boxplot(AccuracyBoxPlot, boxprops=boxprops, patch_artist=True, medianprops =medianprops)
        plt.ylim(0.99, 1.001)
        ## change outline color, fill color and linewidth of the boxes
        
        # Save the figure
        figName = "%s/%s_bestParamValidScore.png" % (algo, algo)
        fig.savefig(figName, bbox_inches='tight', dpi=600)
        count += 1
if __name__ == '__main__':
    
    main()