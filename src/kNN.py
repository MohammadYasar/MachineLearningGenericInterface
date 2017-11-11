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
        
        # sweeping for best method with cross validation
        self.kSweep = [3, 5, 7, 9]
        self.sweepingList = []
        
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
        
    def makeSweepingList(self, kSweep=None):
        """ making a list with all combinations of sweeping parameters
        """
        if kSweep is None:
            self.kSweep = kSweep
        self.sweepingList =  [[i] for i in self.kSweep]
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
        self.makeSweepingList(self.kSweep)
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
                self.loadParametersFromList(params=params )
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
            print('Best model for fold #%d is k=%d with Val. Accu %0.3f \
                  and Test Accu. %0.3f\n' %(foldNo, \
                          bestParams[0], bestValAcc, testaccuracy))
            foldNo += 1

