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
   
    def __init__(self, loggingLevel = logging.INFO, enableLoggingTime = False):
        # randomForrest class constructor
        self.n_estimators = 50
        self.oob_score = True
        self.n_jobs = 4
        super(randomForrest, self).__init__(loggingLevel, enableLoggingTime)
        
        # sweeping for best method with cross validation
        self.n_estimatorsSweep = [51, 75, 101]
        self.sweepingList = []
        
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
    
    
    def makeSweepingList(self, n_estimatorsSweep=None):
        """ making a list with all combinations of sweeping parameters
        """
        if n_estimatorsSweep is None:
            self.n_estimatorsSweep = n_estimatorsSweep
        self.sweepingList =  [[i] for i in self.n_estimatorsSweep]
        return self.sweepingList 
        
    def loadParametersFromList(self, params=[3]):
        """ override model parameters for the object from params
        """
        self.n_neighbors = params[0]
    
    def doubleCrossValidate(self, pfeatures, pClass, nFold=5, pModel=None, 
                      scoring='accuracy'):
        """function for cross validation
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel 
        
        bestParamList=[]
        ValAccuList=[]
        TestAccuList = []
        self.makeSweepingList(self.n_estimatorsSweep)
        # indexes for train and test 
        pKF = self.getKFold(pfeatures, nFold=nFold)
        foldNo = 1
        print( 'Double cross validation with fold %d started ...\n' %(nFold) )
        # folds loop
        for train_index, test_index in pKF.split( pfeatures ):
            #print train_index
            #print test_index
            
            pFeatureTrain = pfeatures[train_index]
            pFeatureTest = pfeatures[test_index]
            pClassTrain= pClass[train_index]
            pClassTest= pClass[test_index] 
            
            bestAccuracy = -1
            # param sweeping list loop
            
            bestModel = []
            for params in self.sweepingList:
                # loading parameters from sweeping list
                self.loadParametersFromList(params=params)
                # loading new model with definite parameters
                self.loadModel()
                
                accuracy, accu_mean, std = self.singleCrossValidate( \
                                                    pFeatureTrain, pClassTrain,
                                                    nFold=5)
                #print accu_mean
                if accu_mean > bestAccuracy:
                    bestValAcc = accu_mean
                    bestParams = params
                    #bestModel = self.model
                    self.saveModel(fileName='best_kNN')

                    
            # loading best model through inner cross validation
            self.loadModel()
            self.trainModel( pFeatureTrain , pClassTrain)
            #print(self.model)
            classPred = self.testModel(pFeatureTest)
            #metrices
            testaccuracy, matConf, matCohenKappa, \
            strClassificationReport = self.getMetrics( classTest = pClassTest, 
                                                 classPred = classPred,
                                                 boolPrint = False)
            # cmap figure generation for confusion matrix
#            self.printConfusionMatrix( matConf )
            printstr1 = "Best model for fold #%d is n_estimator=%d with \n\t" \
                            % (foldNo, bestParams[0])
            printstr2 = "Val. Accu %0.3f\n\t" % ( bestValAcc )
            printstr3 = "Test Accu. %0.3f\n" % ( testaccuracy)
            print printstr1 + printstr2 + printstr3
            foldNo += 1