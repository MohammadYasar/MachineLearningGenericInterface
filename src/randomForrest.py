#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  10 09:05:26 2017

@author: Mirza Elahi
"""
from predictor import predictor
from sklearn.ensemble import RandomForestClassifier
import logging

class randomForrest( predictor ): 
   
    def __init__(self, loggingLevel = logging.DEBUG, enableLoggingTime = False):
        # randomForrest class constructor
        self.n_estimators = 50
        self.oob_score = True
        self.n_jobs = 4
        super(randomForrest, self).__init__(loggingLevel, enableLoggingTime)
        
    def toString(self):
        """ Print parameters of current model
        """
        pStr = "Current model:\n\tRandom Forest model with \n\t\tNo of estimator = %d \
            \n\t\tOOB Score = %d\n\t\tNo of jobs = %d\n" % (self.n_estimators, \
                                self.oob_score, self.n_jobs)
        return pStr
        
    def getModel(self, n_estimators=None, oob_score=None, n_jobs=None):
        """ Temporary model generation
        """
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if oob_score is not None:
            self.oob_score = oob_score    
        if n_jobs is not None:
            self.n_jobs = n_jobs 
        pModel = RandomForestClassifier(n_estimators = self.n_estimators, 
                                            oob_score = self.oob_score, 
                                            n_jobs = self.n_jobs)
        return pModel
    
    def loadModel(self, n_estimators=None, oob_score=None, n_jobs=None):
        """ load internal model
        """
        if n_estimators is not None:
            self.n_estimators = n_estimators
        if oob_score is not None:
            self.oob_score = oob_score    
        if n_jobs is not None:
            self.n_jobs = n_jobs 
            
        self.model = []
        self.model = self.getModel(n_estimators=self.n_estimators, 
                                   oob_score=self.oob_score, 
                                   n_jobs=self.n_jobs)
