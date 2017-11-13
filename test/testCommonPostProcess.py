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

def myBoxPlot(data, algo, figName, ylim=[0.99, 1.001],
              xlabel='Outer Fold No.', ylabel='Accuracy', title=None):
    fig = plt.figure(figsize=(8, 4))
    
    # Create an axes instance
    ax = fig.add_subplot(111)
    boxprops = dict(linewidth=3, color='#3852a3')
    medianprops = dict(linestyle='-', linewidth=2.5, color='#ed1e23')
    # Create the boxplot
    plt.boxplot(data, boxprops=boxprops, 
                patch_artist=True, medianprops =medianprops)
    plt.ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=14)  
    ax.set_ylabel(ylabel, fontsize=14) 
    if title is not None:
       ax.set_title(title, fontsize=14)  
    ## change outline color, fill color and linewidth of the boxes
    
    # Save the figure
    #figName = "%s/%s_bestParamValidScore.png" % (algo, algo)
    fig.savefig(figName, bbox_inches='tight', dpi=600)
 
def myPlot(data, algo, figName, ylim=[0.99, 1.001], xlim=[0.5, 5.5],
              xlabel='Outer Fold No.', ylabel='Accuracy', title=None):
    fig = plt.figure(figsize=(8, 4))
    x = np.arange(1, data.size+1)
    # Create an axes instance
    ax = fig.add_subplot(111)
    boxprops = dict(linewidth=3, color='#3852a3')
    medianprops = dict(linestyle='-', linewidth=2.5, color='#ed1e23')
    # Create the boxplot
    plt.plot(x, data, ls='', marker='s', markersize=10, color='#ed1e23')
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=14)  
    ax.set_ylabel(ylabel, fontsize=14) 
    ax.set_xticks(x)
    if title is not None:
       ax.set_title(title, fontsize=14)  
    ## change outline color, fill color and linewidth of the boxes
    
    # Save the figure
    #figName = "%s/%s_bestParamValidScore.png" % (algo, algo)
    fig.savefig(figName, bbox_inches='tight', dpi=600)
def printForLatexTable(data, foldNo):
    strPrint = "foldNo: %d " % (foldNo+1)
    for i in range(data.size):
        strPrint = strPrint + "& %0.8f " %( data[i] )
        
    print(strPrint+'\\\\ \n')
        
def printForLatexTableValidTest(valid, test):
    print("Valid/Test ---")
    for i in range(valid.size):
        strPrint = " %d " % (i+1)
        strPrint = strPrint + "& %0.8f & %0.8f " %( valid[i], test[i] )
        strPrint = strPrint + '\\\\ \n\hline'
        print(strPrint)
        
def main():
    
    #processList = ['dT', 'kNN', 'rF']
    processList = ['dT', 'kNN', 'rF', 'adaBoost', 'sVM']
    count = 0
    for algo in processList:
    
        dT = predictor( enableLoggingTime=True )
        fileName = "%s/%sDataFinal" % (algo, algo)
        data = dT.loadVariables(fileName=fileName)
        ValAccuList = data['ValAccuList'] 
        ValStdList = data['ValStdList']
        TestAccuList = data['TestAccuList']
        TestConfList = data['TestConfList']
        bestParamList = data['bestParamList']
        
        # Fold data [OuterFoldNo][ParamListIndex][Accu/Conf][InnerFoldNo]
        OuterInnerFoldData = data['OuterInnerFoldData'] 
        sweepingList = data['sweepingList']
        NoOfParam = len(sweepingList)
        print NoOfParam
        OuterFoldNo = data['OuterFoldNo']
        InnerFoldNo = data['InnerFoldNo']
        bestParamIndexInSwpList = data['bestParamIndexInSwpList']
  
        print( "Fold no - Outer %d, Inner %d\n" % (OuterFoldNo, InnerFoldNo) )
        #print TestAccuList
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
            bestParamConf = np.array( FoldData[ bestParamIndexInSwpList[outI] ][ 1 ] )
            print( "\tBest param Avg. Valid. Accu. = %0.8f\n"  % ( np.mean(bestParamAccu) ) )
            print( "\tBest param Valid. Accu. = \t\t")
            print bestParamAccu
            
            # mean of accuracy for all param in outer fold
            FoldAllAvgAccuracy = np.ndarray(shape=(NoOfParam), dtype=float )
            for paramI in range(NoOfParam):
                paramAllAccuracy = np.array( FoldData[paramI][0] )
                paramMeanAccuracy = paramAllAccuracy.mean()
                FoldAllAvgAccuracy[paramI] = paramMeanAccuracy
            printForLatexTable(FoldAllAvgAccuracy, outI)
            for inn in range( InnerFoldNo ):
                fileName = "%s/O%d_I%d_conf" % (algo, outI, inn)
                dT.printConfusionMatrix( bestParamConf[inn], filename=fileName )
            fileName = "%s/Test_O%d_conf" % (algo, outI)
            dT.printConfusionMatrix( TestConfList[outI], filename=fileName )
        printForLatexTableValidTest(ValAccuList, TestAccuList)   
        print ValAccuList
        #print AccuracyBoxPlot
        # Create a figure instance
        figName = "%s/%s_bestParamValidScore.png" % (algo, algo)
        myBoxPlot(AccuracyBoxPlot, algo, figName, ylim=[0.99, 1.001],
                  title='Accuracy with optimal params in Validation Set' )
        figName = "%s/%s_testingScore.png" % (algo, algo)
        myPlot(TestAccuList, algo, figName, ylim=[0.992, 1.001],
              xlabel='Outer Fold No.', ylabel='Accuracy', 
              title='Accuracy with optimal params in Testing Set')
        count += 1
if __name__ == '__main__':
    
    main()