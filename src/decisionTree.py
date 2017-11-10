#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  10 09:05:26 2017

@author: Mirza Elahi
"""
from predictor import predictor
from sklearn import tree
import logging

class decisionTree( predictor ): 
   
    def __init__(self, loggingLevel = logging.INFO, enableLoggingTime = False):
        # randomForrest class constructor
        super(decisionTree, self).__init__(loggingLevel, enableLoggingTime)

    def toString(self):
        """ Print parameters of current model
        """
        pStr = "Current model:\n\tDecision tree model\n"
        return pStr
        
    def getModel(self):
        """ Temporary model generation
        """
        pModel = tree.DecisionTreeClassifier()
        return pModel
    
    def loadModel(self):
        """ load internal model
        """
        self.model = []
        self.model = self.getModel()