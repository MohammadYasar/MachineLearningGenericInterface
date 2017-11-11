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
    ckNN.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 30808)
    ckNN.dataConvertToNumpy()
    # do double cross
    ckNN.doubleCrossValidate(ckNN.featureNumpy, ckNN.ClassNumpy)
    
if __name__ == '__main__':
    
    main()