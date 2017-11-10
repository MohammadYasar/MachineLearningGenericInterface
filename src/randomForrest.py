#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 09:05:26 2017

@author: Shabnam Wahed, Mirza Elahi
"""
from predictor import predictor
from sklearn.ensemble import RandomForestClassifier
import logging

class randomForrest( predictor ): 
   
    def __init__(self, loggingLevel = logging.DEBUG, enableLoggingTime = False):
        # randomForrest class constructor
        self.n_estimators = 100
        self.oob_score = True
        self.n_jobs = 4
        super(randomForrest, self).__init__(loggingLevel, enableLoggingTime)
        
    def trainModel( self, featureTrain, classTrain):
        """overriding virtual function for training
        """
        super(randomForrest, self).trainModel(None, None)
        self.model = RandomForestClassifier(n_estimators = self.n_estimators, 
                                            oob_score = self.oob_score, 
                                            n_jobs = self.n_jobs)
        self.model.fit(featureTrain, classTrain)
        return self.model
    
    def testModel( self, featureTest):
        """overriding virtual function for testing
        """
        super(randomForrest, self).testModel(None, None)
        classPred = self.model.predict( featureTest )
    
        return classPred


