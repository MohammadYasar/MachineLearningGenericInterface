#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  10 09:05:26 2017

@author: Mirza Elahi
"""
from predictor import predictor
from sklearn import tree
import logging
import numpy as np

class decisionTree( predictor ): 
   
    def __init__(self, loggingLevel = logging.INFO, enableLoggingTime = False):
        # randomForrest class constructor
        super(decisionTree, self).__init__(loggingLevel, enableLoggingTime)
        self.maxDepth = None
        # sweeping for best method with cross validation
        self.maxDepthSweep = [31, 51, None]
        self.sweepingList = []

    def toString(self):
        """ Print parameters of current model
        """
        pStr = "Current model:\n\tDecision tree model\n"
        return pStr
        
    def getModel(self, maxDepth=None):
        """ Temporary model generation
        """
        if maxDepth is not None:
            self.maxDepth = maxDepth
        pModel = tree.DecisionTreeClassifier(max_depth=maxDepth)
        return pModel
    
    def loadModel(self, maxDepth=None):
        """ load internal model
        """
        if maxDepth is not None:
            self.maxDepth = maxDepth
        self.model = []
        self.model = self.getModel(self.maxDepth)
        
    def makeSweepingList(self, maxDepthSweep=None):
        """ making a list with all combinations of sweeping parameters
        """
        if maxDepthSweep is not None:
            self.maxDepthSweep = maxDepthSweep
        self.sweepingList =  [[i] for i in self.maxDepthSweep]
        return self.sweepingList 
        
    def loadParametersFromList(self, params=[3]):
        """ override model parameters for the object from params
        """
        self.maxDepth = params[0]
        
    def doubleCrossValidate(self, pfeatures, pClass, nFoldOuter=5, 
                            nFoldInner=4, fileName=None, pModel=None, scoring='accuracy'):
        """function for cross validation
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel 
        
        bestParamList=[]
        ValAccuList=[]
        ValStdList = []
        TestAccuList = []
        self.makeSweepingList(self.maxDepthSweep)
        # indexes for train and test 
        pKF = self.getKFold(pfeatures, nFold=nFoldOuter)
        foldNo = 1
        print( 'Double cross validation with fold %d started ...\n' %(nFoldOuter) )
        OuterInnerFoldData = []
        # folds loop
        for train_index, test_index in pKF.split( pfeatures ):
            #print train_index
            #print test_index
            
            pFeatureTrain = pfeatures[train_index]
            pFeatureTest = pfeatures[test_index]
            pClassTrain= pClass[train_index]
            pClassTest= pClass[test_index] 
            
            bestValAcc = -1
            # param sweeping list loop
            eachInnerFoldData = []
            for params in self.sweepingList:
                # loading parameters from sweeping list
                self.loadParametersFromList(params=params)
                # loading new model with definite parameters
                self.loadModel()
                
                accuracy, accu_mean, std, conf = self.mySingleCrossValidate( \
                                                    pFeatureTrain, pClassTrain,
                                                    nFold=nFoldInner)
                
                
                #print accu_mean
                if accu_mean > bestValAcc:
                    bestValAcc = accu_mean
                    bestValStd = std
                    bestParams = params
                    #bestModel = self.model
                    self.saveModel(fileName='best_DT')
                eachInnerFoldData.append( [accuracy, conf] )
                    
            # loading best model through inner cross validation
            OuterInnerFoldData.append(eachInnerFoldData)
            self.loadSavedModel('best_DT')
            self.trainModel( pFeatureTrain , pClassTrain)
            #print(self.model)
            classPred = self.testModel(pFeatureTest)
            #metrices
            testaccuracy, avgPrecScore, matConf, matCohenKappa, \
            strClassificationReport = self.getMetrics( classTest = pClassTest, 
                                                 classPred = classPred,
                                                 boolPrint = False)
            # cmap figure generation for confusion matrix
#            self.printConfusionMatrix( matConf )
            a = bestParams[0] if bestParams[0] is not None else np.nan
            printstr1 = "Best model for fold #%d is max_depth=%0.0f with \n\t" \
                            % (foldNo, a)
            printstr2 = "Val. Accu %0.5f\n\t" % ( bestValAcc )
            printstr3 = "Test Accu. %0.5f\n" % ( testaccuracy)
            print printstr1 + printstr2 + printstr3
            
            ValAccuList.append(bestValAcc)
            TestAccuList.append(testaccuracy)
            ValStdList.append(bestValStd)
            bestParamList.append(bestParams)
            foldNo += 1
#        OuterInnerFoldData.append(self.sweepingList)
#        OuterInnerFoldData.append([nFoldOuter, nFoldInner])
#        OuterInnerFoldData.append(bestParamList)
        if fileName is not None:
            # OuterInnerFoldData 
            #           [OuterFoldNo][ParamListIndex][Accu/Conf][InnerFoldNo]
            self.saveDoubleCrossValidData(fileName=fileName, 
                                     ValAccuList = ValAccuList, 
                                     ValStdList = bestParamList,
                                     TestAccuList = TestAccuList, 
                                     bestParamList = bestParamList, 
                                     OuterInnerFoldData= OuterInnerFoldData, 
                                     sweepingList = self.sweepingList,
                                     OuterFoldNo = nFoldOuter, 
                                     InnerFoldNo = nFoldInner)
        
        return np.array(ValAccuList), np.array(ValStdList), \
                np.array(TestAccuList), bestParamList, OuterInnerFoldData