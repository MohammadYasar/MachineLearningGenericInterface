#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  10 09:05:26 2017

@author: Mirza Elahi
"""
from predictor import predictor, scoring, algo
from sklearn import svm
import logging
import numpy as np


class sVM( predictor ): 
       
    def __init__(self, loggingLevel = logging.INFO, enableLoggingTime = False):
        # kNN class constructor
        super(sVM, self).__init__(loggingLevel, enableLoggingTime)
        self.kernel = 'linear'
        self.C=1
        self.gamma='auto'
        self.max_iter = 50
        
        # sweeping for best method with cross validation
        self.kernelSweep = ['linear', 'poly', 'rbf']
        self.CSweep = [1, 100, 1000]
        self.gammaSweep = ['auto', 10, 100] 
        
    def toString(self):
        """ Print parameters of current model
        """
        pStr = "Current model:\n\tSVM model with \n\t\tkernel = %s\n\t\tC = %d\n\t\tgamma = %s\n" \
                        % (self.kernel, self.C, str(self.gamma))
        return pStr
        
    def getModel(self, kernel=None, C=None, gamma=None):
        """ Temporary model generation
        """
        if kernel is not None:
            self.kernel = kernel
        if C is not None:
            self.C = C
        if gamma is not None:
            self.gamma = gamma
        pModel = svm.SVC(kernel=self.kernel, C=self.C, gamma=self.gamma, 
                         max_iter=self.max_iter)
        return pModel
    
    def loadModel(self, kernel=None, C=None, gamma=None):
        """ load internal model
        """
        if kernel is not None:
            self.kernel = kernel
        if C is not None:
            self.C = C
        if gamma is not None:
            self.gamma = gamma
        self.model = []
        self.model = self.getModel(kernel=self.kernel, C=self.C, 
                                   gamma=self.gamma)
        
    def makeSweepingList(self, kernelSweep=None, CSweep=None, gammaSweep=None):
        """ making a list with all combinations of sweeping parameters
        """
        if kernelSweep is not None:
            self.kernelSweep = kernelSweep
        if CSweep is not None:
            self.CSweep = CSweep
        if gammaSweep is not None:
            self.gammaSweep = gammaSweep
        self.sweepingList =  [[i, j, k] for i in self.kernelSweep \
                              for j in self.CSweep for k in self.gammaSweep]
        return self.sweepingList 
        
    def loadParametersFromList(self, params=['linear', 1, 'auto']):
        """ override model parameters for the object from params
        """
        self.kernel = params[0]
        self.C = params[1]
        self.gamma = params[2]
    
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
        self.makeSweepingList(self.kernelSweep, self.CSweep, self.gammaSweep)
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
                                                    isStratified=isStratified)
                scoreMean = score.mean()
                scoreStd = score.std()
                
                #print params
                print scoreMean
                if scoreMean > bestScoreMean:
                    bestScoreMean = scoreMean
                    bestScoreStd = scoreStd
                    bestParams = params
                    #bestModel = self.model
                    self.saveModel(fileName='best_svm')
                eachInnerFoldData.append( [score, accuracy, mccs, conf] )
            OuterInnerFoldData.append(eachInnerFoldData) 
            # loading best model through inner cross validation
            # model in 'best_svm'
            self.loadSavedModel(fileName='best_svm')
            self.trainModel( pFeatureTrain , pClassTrain)
            #print(self.model)
            # test model
            classPred = self.testModel(pFeatureTest)
            #metrices
            testScore, testaccuracy, avgPrecScore, matConf, matCohenKappa, \
            strClassificationReport, mcc = self.getMetrics( classTest = pClassTest, 
                                                 classPred = classPred,
                                                 scoring=scoring,
                                                 boolPrint = False)
            printstr1 = "Best model for fold #%d is kernel=%s, C=%d, gamma=%s with \n\t" \
                            % ( foldNo, bestParams[0], bestParams[1], str(bestParams[2]) )
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
                                     algorithm = algo.SVM )
        
        return np.array(ValScoreList), np.array(ValScoreStdList), \
                np.array(TestScoreList), bestParamList, OuterInnerFoldData