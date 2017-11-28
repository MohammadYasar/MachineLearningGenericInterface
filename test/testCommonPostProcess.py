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
import matplotlib as mpl
import numpy as np

def myBoxPlot(data, algo, figName, ylim=[0.99, 1.001],
              xlabel='Outer Fold No.', ylabel='Accuracy', title=None):
    """ function for Box plot 
        data in columns for each box
    """
    fig = plt.figure(figsize=(8, 4))
    
    # Create an axes instance
    ax = fig.add_subplot(111)
    boxprops = dict(linewidth=3, color='#3852a3')
    medianprops = dict(linestyle='-', linewidth=2.5, color='#ed1e23')
    label_size = 14
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size
    # Create the boxplot
    plt.boxplot(data, boxprops=boxprops, 
                patch_artist=True, medianprops =medianprops,
                widths = 0.2)
    plt.ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=label_size+2)  
    ax.set_ylabel(ylabel, fontsize=label_size+2) 
    if title is not None:
       ax.set_title(title, fontsize=14)  
    ## change outline color, fill color and linewidth of the boxes
    
    # Save the figure
    #figName = "%s/%s_bestParamValidScore.png" % (algo, algo)
    fig.savefig(figName, bbox_inches='tight', dpi=600)
 
def myPlot(data, algo, figName, ylim=[0.99, 1.001], xlim=[0.5, 5.5],
              xlabel='Outer Fold No.', ylabel='Accuracy', title=None):
    """ function for Box plot 
        data in columns for each box
    """
    fig = plt.figure(figsize=(8, 4))
    x = np.arange(1, data.size+1)
    # Create an axes instance
    ax = fig.add_subplot(111)
    
    plt.plot(x, data, ls='', marker='s', markersize=10, color='#ed1e23')
    # mean value
    plt.plot(np.array(xlim), np.array([np.mean(data), np.mean(data)]), 
             ls='--', lw=1, color='#ed1e23')
    plt.xlim(xlim)
    plt.ylim(ylim)
    ax.set_xlabel(xlabel, fontsize=14)  
    ax.set_ylabel(ylabel, fontsize=14) 
    ax.set_xticks(x)
    if title is not None:
       ax.set_title(title, fontsize=14)
    fig.savefig(figName, bbox_inches='tight', dpi=600)
    
def printForLatexTable(data, foldNo):
    """ function for printing LaTex format table details of each validation
    """
    strPrint = "foldNo: %d " % (foldNo+1)
    for i in range(data.size):
        strPrint = strPrint + "& %0.8f " %( data[i] )
        
    print(strPrint+'\\\\ \n')
        
def printForLatexTableValidTest(valid, test):
    """ function for printing LaTex format table avg. valid and test score only
    """
    print("Valid/Test ---")
    for i in range(valid.size):
        strPrint = " %d " % (i+1)
        strPrint = strPrint + "& %0.8f & %0.8f " %( valid[i], test[i] )
        strPrint = strPrint + '\\\\ \n\hline'
        print(strPrint)
    strPrint = "mean & %0.8f & %0.8f\\\\ \n\hline" % (valid.mean(), test.mean())
    print(strPrint)    
def main():
    
    #processList = ['xgBoost']
    printConfMat = False
    processList = ['dT', 'kNN', 'rF', 'adaBoost', 'sVM', 'xgBoost']
    count = 0
    for algo in processList:
    
        dT = predictor( enableLoggingTime=True )
        fileName = "%s/%sDataProj" % (algo, algo)
        data = dT.loadVariables(fileName=fileName)
        ValScoreList = data['ValScoreList'] 
        ValScoreStdList = data['ValScoreStdList']
        TestScoreList = data['TestScoreList']
        TestConfList = data['TestConfList']
        bestParamList = data['bestParamList']
        scoring = data['scoring']
        scoring = scoring.name 
        
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
        
        ScoreBoxPlot = np.ndarray(shape=(InnerFoldNo, OuterFoldNo), dtype=float )
        # Fold 1
        for outI in range( OuterFoldNo ):
            print( "Fold #%d\n" % (outI) )
            FoldData = OuterInnerFoldData[outI]
            bestParamScore = np.array( FoldData[bestParamIndexInSwpList[outI]][0] )
            ScoreBoxPlot[:, outI] = bestParamScore
            bestParamConf = np.array( FoldData[ bestParamIndexInSwpList[outI] ][ 3 ] )
            print( "\tBest param Avg. Valid. Score = %0.8f\n"  % ( np.mean(bestParamScore) ) )
            print( "\tBest param Valid. Score. = \t\t")
            print bestParamScore
            
            # mean of accuracy for all param in outer fold
            FoldAllAvgScore = np.ndarray(shape=(NoOfParam), dtype=float )
            for paramI in range(NoOfParam):
                paramAllScore = np.array( FoldData[paramI][0] )
                paramMeanScore = paramAllScore.mean()
                FoldAllAvgScore[paramI] = paramMeanScore
            printForLatexTable(FoldAllAvgScore, outI)
            if printConfMat:
                for inn in range( InnerFoldNo ):
                    fileName = "%s/O%d_I%d_conf" % (algo, outI, inn)
                    dT.printConfusionMatrix( bestParamConf[inn], filename=fileName )
                fileName = "%s/Test_O%d_conf" % (algo, outI)
                dT.printConfusionMatrix( TestConfList[outI], filename=fileName )
        printForLatexTableValidTest(ValScoreList, TestScoreList)   
        print ValScoreList
        #print AccuracyBoxPlot
        # Create a figure instance
        figName = "%s/%s_bestParamValidScore.eps" % (algo, algo)
        myBoxPlot(ScoreBoxPlot, algo, figName, ylim=[0, 1.0],
                  title='%s with optimal params in Validation Set' % (scoring), 
                  ylabel='MCC' )
        figName = "%s/%s_testingScore.eps" % (algo, algo)
        myPlot(TestScoreList, algo, figName, ylim=[0, 1.0],
              xlabel='Outer Fold No.', ylabel='MCC', 
              title='%s with optimal params in Testing Set' % (scoring) )
        count += 1
if __name__ == '__main__':
    
    main()