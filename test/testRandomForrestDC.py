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
    rf.loadData( fileName = '../data/creditcard.csv', feaRowEnd = 20008)
    #converting to numpy
    rf.dataConvertToNumpy()
    # double cross validation
    rf.doubleCrossValidate(rf.featureNumpy, rf.ClassNumpy)
    
if __name__ == '__main__': 
    main()