#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  10 09:05:26 2017

@author: Mirza Elahi
"""
from predictor import predictor, scoring, algo
from sklearn.ensemble import AdaBoostClassifier
import logging
import numpy as np

class adaBoost( predictor ): 
   
    def __init__(self, loggingLevel = logging.INFO, enableLoggingTime = False):
        # randomForrest class constructor
        self.n_estimators = 51
        super(adaBoost, self).__init__(loggingLevel, enableLoggingTime)
        
        # sweeping for best method with cross validation
        self.n_estimatorsSweep = [31, 51, 71]
        self.sweepingList = []
        
    def toString(self):
        """ Print parameters of current model
        """
        pStr = "Current model:\n\tAdaboost model with DecisionTreeClassifier model with \n\t\tNo of estimator = %d\n" % (self.n_estimators)
        return pStr
        
    def getModel(self, n_estimators=None):
        """ Temporary model generation
        """
        if n_estimators is not None:
            self.n_estimators = n_estimators
        pModel = AdaBoostClassifier(n_estimators = self.n_estimators)
        return pModel
    
    def loadModel(self, n_estimators=None):
        """ load internal model
        """
        if n_estimators is not None:
            self.n_estimators = n_estimators
            
        self.model = []
        self.model = self.getModel(n_estimators=self.n_estimators)
    
    
    def makeSweepingList(self, n_estimatorsSweep=None):
        """ making a list with all combinations of sweeping parameters
        """
        if n_estimatorsSweep is not None:
            self.n_estimatorsSweep = n_estimatorsSweep
        self.sweepingList =  [[i] for i in self.n_estimatorsSweep]
        return self.sweepingList 
        
    def loadParametersFromList(self, params=[3]):
        """ override model parameters for the object from params
        """
        self.n_estimators = params[0]
            
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
        self.makeSweepingList( self.n_estimatorsSweep )
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
                self.loadParametersFromList( params=params )
                # loading new model with definite parameters
                self.loadModel()
                
                score, \
                accuracy, \
                conf, \
                mccs = self.mySingleCrossValidate( pFeatureTrain, pClassTrain,
                                                    scoring=scoring,
                                                    nFold=nFoldInner,
                                                    isStratified=isStratified )
                scoreMean = score.mean()
                scoreStd = score.std()
                
                #print params
                print scoreMean
                if scoreMean > bestScoreMean:
                    bestScoreMean = scoreMean
                    bestScoreStd = scoreStd
                    bestParams = params
                    #bestModel = self.model
                    self.saveModel(fileName='best_ada')
                eachInnerFoldData.append( [score, accuracy, mccs, conf] )
            OuterInnerFoldData.append(eachInnerFoldData)
                    
            # loading best model through inner cross validation
            self.loadSavedModel(fileName='best_ada')
            self.trainModel(pFeatureTrain , pClassTrain)
            # test model
            classPred = self.testModel(pFeatureTest)
            #metrices
            testScore, testaccuracy, avgPrecScore, matConf, matCohenKappa, \
            strClassificationReport, mcc = self.getMetrics( classTest = pClassTest, 
                                                 classPred = classPred,
                                                 scoring=scoring,
                                                 boolPrint = False)
            
            printstr1 = "Best model for fold #%d is n_estimator=%d with \n\t" \
                            % (foldNo, bestParams[0])
            printstr2 = "Val. Score %0.5f\n\t" % ( bestScoreMean )
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
                                     algorithm = algo.adaBoost )
        
        return np.array(ValScoreList), np.array(ValScoreStdList), \
                np.array(TestScoreList), bestParamList, OuterInnerFoldData