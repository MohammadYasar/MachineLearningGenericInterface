#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  10 09:05:26 2017

@author: Mirza Elahi
"""
from predictor import predictor
from sklearn.neighbors import KNeighborsClassifier
import logging

class kNN( predictor ): 
       
    def __init__(self, loggingLevel = logging.INFO, enableLoggingTime = False):
        # kNN class constructor
        super(kNN, self).__init__(loggingLevel, enableLoggingTime)
        self.n_neighbors = 3
        
    def toString(self):
        """ Print parameters of current model
        """
        pStr = "Current model:\n\tk Nearest Neighbour model with \n\t\tk = %d\n" \
                        % (self.n_neighbors)
        return pStr
        
    def getModel(self, n_neighbors=None):
        """ Temporary model generation
        """
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        pModel = KNeighborsClassifier(n_neighbors=self.n_neighbors)
        return pModel
    
    def loadModel(self, n_neighbors=None):
        """ load internal model
        """
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        self.model = []
        self.model = self.getModel(n_neighbors=self.n_neighbors)
        


