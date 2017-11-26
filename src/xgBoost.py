#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  10 09:05:26 2017

@author: Mirza Elahi
"""
from predictor import predictor, scoring, algo
from xgboost import XGBClassifier
import logging
import numpy as np

class xgBoost( predictor ): 
   
    def __init__(self, loggingLevel = logging.INFO, enableLoggingTime = False):
        # xgBoost class constructor
        super(xgBoost, self).__init__(loggingLevel, enableLoggingTime)
        self.n_estimators = 100
        self.num_round = 10
        self.n_thread = 4
        self.eval_metric = 'auc'
        self.eta = 1
        self.silent = True
        self.objective = 'binary:logistic'
        self.max_depth = 6
        self.learning_rate = 0.1
        
        # sweeping for best method with cross validation
        self.max_depthSweep = [6, 10]
        self.n_estimatorsSweep = [51, 71, 91]
        self.sweepingList = []
        
    def toString(self):
        """ Print parameters of current model
        """
        pStr = "Current model:\n\txgBoost model with \n\t\tmax depth = %d \
            \n\t\tnum of round = %d\n\t\tNo of threads = %d\n" % (self.max_depth, \
                                self.num_round, self.n_thread)
        return pStr
        
    def getModel(self, max_depth=None, learning_rate=None, n_estimators=None, 
                 objective=None, silent=None):
        """ Temporary model generation
        """
        if max_depth is not None:
            self.max_depth = max_depth
        if learning_rate is not None:
            self.learning_rate = learning_rate 
        if n_estimators is not None:
            self.n_estimators = n_estimators  
        if objective is not None:
            self.objective = objective
        if silent is not None:
            self.silent = silent 
        
        pModel = XGBClassifier(max_depth = self.max_depth,
                               learning_rate = self.learning_rate,
                               n_estimators = self.n_estimators,
                               silent = self.silent,
                               objective = self.objective)
        return pModel
    
    def loadModel(self, max_depth=None, learning_rate=None, n_estimators=None, 
                 objective=None, silent=None):
        """ load internal model
        """
        if max_depth is not None:
            self.max_depth = max_depth
        if learning_rate is not None:
            self.learning_rate = learning_rate 
        if n_estimators is not None:
            self.n_estimators = n_estimators  
        if objective is not None:
            self.objective = objective
        if silent is not None:
            self.silent = silent   
        self.model = []
        self.model = self.getModel(max_depth=self.max_depth,
                                   learning_rate=self.learning_rate,
                                   n_estimators=self.n_estimators,
                                   objective=self.objective,
                                   silent=self.silent)
        
    def trainModel( self, featureTrain, classTrain, pModel=None):
        """overriding virtual function for training
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel
        # training the model
        self.baseLogger.debug('Trainng model ...')
        self.model.fit(featureTrain, classTrain, 
                       eval_metric=self.eval_metric)
        return self.model
    
    def makeSweepingList(self, max_depthSweep=None, n_estimatorsSweep=None):
        """ making a list with all combinations of sweeping parameters
        """
        if max_depthSweep is not None:
            self.max_depthSweep = max_depthSweep
        if n_estimatorsSweep is not None:
            self.n_estimatorsSweep = n_estimatorsSweep
            
        self.sweepingList =  [[i, j] for i in self.max_depthSweep \
                              for j in self.n_estimatorsSweep]
        return self.sweepingList 
        
    def loadParametersFromList(self, params=[6, 51]):
        """ override model parameters for the object from params
        """
        self.max_depth = params[0]
        self.n_estimators = params[1]
        
    def doubleCrossValidate(self, pfeatures, pClass, nFoldOuter=5, 
                            nFoldInner=4, fileName=None, pModel=None, 
                            scoring=scoring.ACCURACY,
                            isStratified=False):
        """function for cross validation
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel 
        
        bestParamList=[]
        ValScoreList=[]
        ValScoreStdList = []
        TestScoreList = []
        TestConfList = []
        self.makeSweepingList(self.max_depthSweep, self.n_estimatorsSweep)
        # indexes for train and test 
        pKF = self.getKFold(pfeatures, nFold=nFoldOuter, 
                            isStratified=isStratified)
        foldNo = 1
        print( 'Double cross validation with fold %d started ...\n' %(nFoldOuter) )
        OuterInnerFoldData = []
        # folds loop
        for train_index, test_index in pKF.split( pfeatures, pClass ):
            pFeatureTrain = pfeatures[train_index]
            pFeatureTest = pfeatures[test_index]
            pClassTrain= pClass[train_index]
            pClassTest= pClass[test_index] 
            
            bestScoreMean = -1E5
            eachInnerFoldData = []
            # param sweeping list loop
            for params in self.sweepingList:
                # loading parameters from sweeping list
                self.loadParametersFromList(params=params )
                # loading new model with definite parameters
                self.loadModel()
                
                score, \
                accuracy, \
                conf, \
                mccs = self.mySingleCrossValidate( pFeatureTrain, pClassTrain,
                                                    scoring=scoring,
                                                    nFold=nFoldInner,
                                                    isStratified=isStratified)
                scoreMean = score.mean()
                scoreStd = score.std()
                
                print scoreMean
                if scoreMean > bestScoreMean:
                    bestScoreMean = scoreMean
                    bestScoreStd = scoreStd
                    bestParams = params
                    #bestModel = self.model
                    self.saveModel(fileName='best_xgBoost')
                eachInnerFoldData.append( [score, accuracy, mccs, conf] )
            
            OuterInnerFoldData.append(eachInnerFoldData) 
            # loading best model through inner cross validation from the saved
            # model in 'best_RF'
            self.loadSavedModel(fileName='best_xgBoost')
            self.trainModel( pFeatureTrain , pClassTrain)
            # test model
            classPred = self.testModel(pFeatureTest)
            # metrices
            testScore, testaccuracy, avgPrecScore, matConf, matCohenKappa, \
            strClassificationReport, mcc = self.getMetrics( classTest = pClassTest, 
                                                 classPred = classPred,
                                                 scoring=scoring,
                                                 boolPrint = False)
            # print logs
            printstr1 = "Best model for fold #%d is max_depth=%d and n_estimator=%d with \n\t" \
                            % (foldNo, bestParams[0], bestParams[1])
            printstr2 = "Avg. Val. Score %0.5f\n\t" % ( bestScoreMean )
            printstr3 = "Test Score. %0.5f\n" % ( testScore )
            print printstr1 + printstr2 + printstr3
            
            ValScoreList.append(bestScoreMean)
            ValScoreStdList.append(bestScoreStd)
            TestScoreList.append(testScore)
            TestConfList.append(matConf)
            bestParamList.append(bestParams)
            foldNo += 1
        if fileName is not None:
            # OuterInnerFoldData 
            # [OuterFoldNo][ParamListIndex][Score, Accu, MCC, Conf][InnerFoldNo]
            self.saveDoubleCrossValidData( fileName=fileName, 
                                     ValScoreList = ValScoreList, 
                                     ValScoreStdList = ValScoreStdList,
                                     TestScoreList = TestScoreList,
                                     TestConfList = TestConfList,
                                     bestParamList = bestParamList, 
                                     OuterInnerFoldData= OuterInnerFoldData, 
                                     sweepingList = self.sweepingList,
                                     OuterFoldNo = nFoldOuter, 
                                     InnerFoldNo = nFoldInner,
                                     scoring = scoring,
                                     algorithm = algo.xgBoost )
        
        return np.array(ValScoreList), np.array(ValScoreStdList), \
                np.array(TestScoreList), bestParamList, OuterInnerFoldData