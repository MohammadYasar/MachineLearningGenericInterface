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

def myBoxPlot(data, algo, figName, xticks, ylim=[0.99, 1.001],
              xlabel='Algorithms', ylabel='Accuracy', title=None):
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
    ax.set_xlabel(xlabel, fontsize=16)  
    ax.set_ylabel(ylabel, fontsize=16) 
    ax.set_xticklabels(xticks, rotation=45)
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
    
    processList = ['sVM']
    #processList = ['dT', 'kNN', 'rF', 'adaBoost', 'sVM']
    count = 0
    ScoreValidBoxPlot = np.ndarray(shape=(5, len(processList)), dtype=float )
    ScoreTestBoxPlot = np.ndarray(shape=(5, len(processList)), dtype=float )
    algoNames = []
    for algo in processList:
    
        dT = predictor( enableLoggingTime=True )
        fileName = "%s/%sData" % (algo, algo)
        data = dT.loadVariables(fileName=fileName)
        ValScoreList = data['ValScoreList'] 
        TestScoreList = data['TestScoreList']
        algoName = data['algorithm']
        ScoreValidBoxPlot[:, count] = ValScoreList
        ScoreTestBoxPlot[:, count] = TestScoreList
        algoNames.append( algoName.name )
        count += 1
        #    print ('%s & %0.2f & %s & %s & %s\\\\ \n \hline' % \
#           (state_ind, X_train_not_normalized[i], tmp_2004, \
#            tmp_pred, tmp_GT))
    figName = "All/validationScoreCompare.png"
    #algoNames = ['DT', 'kNN', 'RF', 'AdaBoost', 'SVM']
    myBoxPlot(ScoreValidBoxPlot, algo, figName, ylim=[0, 1.0],
                  title='Validation MCC with different algorithms',
                  ylabel='MCC', 
                  xticks=algoNames )
    figName = "All/testingScoreCompare.png"
    myBoxPlot(ScoreTestBoxPlot, algo, figName, ylim=[0, 1.0],
                  title='Testing MCC with different algorithms',
                  ylabel='MCC',
                  xticks=algoNames )
    
if __name__ == '__main__':
    
    main()