# -*- coding: utf-8 -*-
"""
This module offers class for general machine learning algorithm 

Author:        Mirza Elahi (me5vp)
Changelog:     2017-10-31 v0.0
"""

import matplotlib
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# oversampling
from imblearn.over_sampling import SMOTE
# machine learning
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
#from math import pi, sqrt, sin, cos, tan
#from sets import Set
import pickle
import sys
import os
import shutil
import operator as op
import time
import scipy.io as sio
import logging
import seaborn as sn
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, matthews_corrcoef
import enum

class scoring(enum.Enum):
    """Enum class for scoring method tracker"""
    # scoring methods
    ACCURACY = 0
    MCC = 1
        
class algo(enum.Enum):
    """Enum class for algorithm type"""
    # algorithms
    DT = 0          # Decision Tree
    kNN = 1         # k nearest neighbor
    RF = 2          # random forest
    adaBoost = 3    # adaBoost
    SVM = 4         # Support Vector Machine

class predictor(object):
    
    def __init__(self, loggingLevel = logging.DEBUG, enableLoggingTime = False):
        """constructor of the class"""
        
        self.dF = []
        self.feature = []
        self.Class = []
        self.featureNumpy = []
        self.ClassNumpy = []
        self.featureNo = []
        self.dataNo = []
        self.featureNames = []
        self.feaRowStart = []
        self.feaRowEnd = []
        
        self.model = []
        
        self.fTrain = []
        self.fTest = []
        self.cTrain = []
        self.cTest = []
        self.baseLogger = []
        self.fileName = []
        
        self.loggingLevel = loggingLevel
        if enableLoggingTime:
            self.loggingFormatter = \
                '[%(asctime)s] %(levelname)s: [%(name)s] %(message)s'
        else:
            self.loggingFormatter = '%(levelname)s:%(message)s'
        self.initLogger()
        self.baseLogger.debug('Initializing machine learning predictor')
        
    def initLogger( self ):
        """Initializing Logger
        """
        # create logger
        self.baseLogger = logging.getLogger(self.__class__.__name__)
        self.baseLogger.handlers = [h for h in self.baseLogger.handlers \
                               if not isinstance(h, logging.StreamHandler)]
        self.baseLogger.setLevel( self.loggingLevel )
        # create file handler which logs even debug messages
        fh = logging.FileHandler('predictor.log')
        fh.setLevel(logging.DEBUG)
        # create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel( self.loggingLevel )
        # create formatter
        formatter = logging.Formatter( self.loggingFormatter,
                                      datefmt='%m/%d/%Y %I:%M:%S %p' )
        # add formatter to ch and fh
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add ch to logger
        self.baseLogger.addHandler(ch)
        #self.baseLogger.addHandler(fh)
        
    def loadData( self, fileName, colClass = 31, colFeaStart = 1, 
                 colFeaEnd = 30, feaRowStart = 2, 
                 feaRowEnd = 284808, delimiter = ',' ):
        """loading data
           @param fileName - name of the csv file
           @param fearowStart - starting row of the feature or Class
           @param feaRowEnd - end row of the feature or Class
           @param colClass - number of the col for Class (indexing starts from 0)
           @param delimitter - default comma
        """
        self.feaRowStart = feaRowStart
        self.feaRowEnd = feaRowEnd
        
        self.dF = pd.read_csv( fileName, delimiter = delimiter)
        #if feaRowStart is not None and feaRowEnd is not None \
               #colFeaStart is not None and colFeaEnd is not None :
        self.feature = self.dF.iloc[feaRowStart-1:feaRowEnd, 
                                    colFeaStart-1:colFeaEnd]
        self.Class = self.dF.iloc[feaRowStart-1:feaRowEnd, colClass-1]
        self.dataNo = feaRowEnd - feaRowStart + 1
        self.featureNo = colFeaEnd - colFeaStart + 1
        
        self.featureNames = self.dF.columns.values.tolist()
        self.featureNames.remove('Class')
        print('Database loaded with: \n \
                  \t Feature No - %d \n \
                  \t Total data points - %d\n' % (self.featureNo, self.dataNo))
        self.dataConvertToNumpy()
    
    def selectImportantFeatures( self, indices ):
        """selecting features based on given column indices
        """
        previousFeatureNo = self.featureNo
        self.featureNames = [self.featureNames[i] for i in indices]
        self.feature = self.feature.loc[:, self.featureNames]
        self.featureNo = len(indices)
        self.dataConvertToNumpy()
        
        print( "Feature dimensionalty reduction:\n\tPrevious Size: %d\n\tCurrent Size: %d" % (previousFeatureNo, self.featureNo) )
       
    def scaleFeature( self, minm=0, maxm=1, copy=True ):
        """scaling features within minm and maxm
        """
        minmax_scale = MinMaxScaler(feature_range=(minm, maxm), copy=copy)
        tempFea = minmax_scale.fit_transform( self.feature )
        self.feature = pd.DataFrame( tempFea, columns=self.featureNames )
        self.dataConvertToNumpy()
        print( "Feature scaling done between %f and %f" % (minm, maxm) )
       
        
        
    def dataConvertToNumpy( self ):
        """Convert feature and Class to numpy, this can also be used for 
           updating
        """
        self.featureNumpy = np.asarray( self.feature )
        self.ClassNumpy = np.asarray( self.Class )
        
    def overSampling( self, feature, Class, random_state = 0 ):
        """utility function for copying unbalanced data for multiple times 
           to balance dataset
           @param feature
           @param Class
           @param random_state - seeding random number
           @return resampled feature
           @return resampled Class
        """
        oversampler = SMOTE(random_state=0)
        feature_resample, Class_resample = oversampler.fit_sample(feature, 
                                                                      Class)
        print("Warning: You are increasing the dataset to balance the data\n")
        return feature_resample, Class_resample
    
    def testTrainSplit(self, feature, Class, test_size = 0.2, 
                           random_state = 0):
        """utility function for splitting in test and train (fixed)
        """
        # training and testing sets
        fTrain, fTest, cTrain, cTest = train_test_split( feature, Class,
                                            test_size = test_size, 
                                            random_state = random_state)
        self.fTrain = fTrain
        self.fTest = fTest
        self.cTrain = cTrain
        self.cTest = cTest
        
        return fTrain, fTest, cTrain, cTest
    
    def getMetrics( self, classTest, classPred, scoring=scoring.ACCURACY, 
                   boolPrint = True):
        """copying unbalanced for multiple times to balance dataset
           @param classTest
           @param classPred
           @return accruacy - one number
           @return matConf - 2x2 matrix for 2 class classifier
           @return matCohenKappa - TODO
           @return strClassificationReport - TODO
           @return mcc - Matthews correlation coefficient
        """
        score = 0
        # accuracy of the model - in one number
        accuracy = accuracy_score(classTest, classPred)
        avgPrecScore = average_precision_score( classTest, classPred )
        # confusion matrix 2x2 matric
        matConf = confusion_matrix(classTest, classPred)
        # cohen Kappa is applicable for unbalanced data
        matCohenKappa = cohen_kappa_score(classTest, classPred)
        # classification report
        strClassificationReport = classification_report(classTest, classPred)
        # Matthews correlation coefficient
        mcc = matthews_corrcoef( classTest, classPred )
        if boolPrint is True:
            print('Accruacy = %f %%\n' % (accuracy*100) )
            print('Classification Report:\n = %s\n' % (strClassificationReport) )
            print('Confusion Matrix:\n' )
            print(matConf)
            print('\n')
            print('Matthews Corr. Coeff. = %f\n' % (mcc) )
        # selecting score for comparison
        if scoring == scoring.ACCURACY:
            score = accuracy
        elif scoring == scoring.MCC:
            score = mcc
            
        return score, accuracy, avgPrecScore, matConf, matCohenKappa, \
            strClassificationReport, mcc
    
    def trainModel( self, featureTrain, classTrain, pModel=None):
        """overriding virtual function for training
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel
        # training the model
        self.baseLogger.debug('Trainng model ...')
        self.model.fit(featureTrain, classTrain)
        return self.model
    
    def testModel( self, featureTest, pModel=None):
        """overriding virtual function for testing
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel 
        self.baseLogger.debug('Testing model ...')
        # predicting with trained model
        classPred = self.model.predict( featureTest )
        return classPred
    
    def printConfusionMatrix( self, confMatrix, 
                             classLabels=["0", "1"],
                             filename='confusionMatrix', dpi=600, cmap=None ):
        
        cmap = cmap or sn.cubehelix_palette(dark=0, light=1, as_cmap=True)
        df_cm = pd.DataFrame(confMatrix, index = [i for i in classLabels],
                  columns = [i for i in classLabels])
        plt.figure(figsize = (4,3.7))
        ax = sn.heatmap(df_cm, annot=True, cmap=cmap, fmt='g')
        for _, spine in ax.spines.items():
            spine.set_visible(True)
        ax.xaxis.tick_top()
        ax.set_xlabel('Predicted', fontsize=14)  
        ax.set_ylabel('Actual', fontsize=14)  
        ax.xaxis.set_label_position('top')
        ax.yaxis.set_label_position('left')
        plt.savefig( filename+".png", dpi=dpi )
        
    def getKFold(self, pfeatures, nFold=5, isStratified=False):
        """function for index of train and test split for nFold
        """
        if isStratified:
            kf = StratifiedKFold( n_splits=nFold )
        else:
            kf = KFold( n_splits=nFold )
        return kf
        
    def singleCrossValidate(self, pfeatures, pClass, nFold=5, pModel=None, 
                      scoring='accuracy'):
        """function for cross validation with built in python lib
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel 
        scores = cross_val_score(self.model, pfeatures, pClass, cv=nFold, 
                                 scoring=scoring)
        return scores, scores.mean(), scores.std()
    
    def mySingleCrossValidate(self, pfeatures, pClass, nFold=5, pModel=None, 
                      scoring=scoring.ACCURACY, isStratified=False):
        """function for cross validation from scratch
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel 
#        scores = cross_val_score(self.model, pfeatures, pClass, cv=nFold, 
#                                 scoring=scoring)
        scores = []
        accuList = []
        mccList = []
        confMatList = []
        pKF = self.getKFold(pfeatures, nFold=nFold, 
                            isStratified=isStratified)
        foldNo = 1
        tempModel = self.model # just for safety
        for train_index, test_index in pKF.split( pfeatures, pClass ):
            pFeatureTrain = pfeatures[train_index]
            pFeatureTest = pfeatures[test_index]
            pClassTrain= pClass[train_index]
            pClassTest= pClass[test_index] 
            self.model = tempModel
            self.trainModel( pFeatureTrain, pClassTrain )
            classPred = self.testModel( pFeatureTest )
            #metrices
            score, accuracy, avgPrecScore, matConf, matCohenKappa, \
            strClassificationReport, mcc = self.getMetrics( pClassTest, 
                                                 classPred,
                                                 scoring = scoring,
                                                 boolPrint = False)
            scores.append( score )
            accuList.append( accuracy )
            confMatList.append( matConf )
            mccList.append( mcc )
            foldNo += 1
        scores = np.array( scores )
        accuracies = np.array( accuList )
        mccs = np.array( mccList )
        return scores, accuracies, confMatList, mccs
    
    def saveModel(self, fileName, pModel=None):
        """saving model to a file for future use
        """
        if pModel is not None:
            self.model = pModel
        self.fileName = fileName + '.sav'
        #pickle.dump(self.model, open(fileName, 'wb')) 
        joblib.dump(self.model, self.fileName)
        self.baseLogger.debug('Saved model in %s' % self.fileName)

    def loadSavedModel(self, fileName=None):
        """loading previously saved model
        """
        fileName = fileName + '.sav'
        if fileName is not None:
            self.fileName = fileName
        #return pickle.load(open(self.fileName, 'rb'))  
        print("Loading saved model from: %s\n" % (self.fileName))
        return joblib.load(self.fileName)
    
    def saveVariables(self, data, fileName):
        """saving variable as dictionary
        """
        fileName = fileName + '.pkl'
#        # Saving the objects:
#        with open(fileName, 'w') as f:  # Python 3: open(..., 'wb')
#            pickle.dump(data, f)
        # Store data (serialize)
        with open(fileName, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def loadVariables(self, fileName):
        """loading variable as dictionary
        """
        fileName = fileName + '.pkl'
        # Getting back the objects:
#        with open(fileName) as f:  # Python 3: open(..., 'rb')
#            data = pickle.load(f)
        # Load data (deserialize)
        with open(fileName, 'rb') as handle:
            data = pickle.load(handle)
        return data
    def getBlankCLass(self):
        """returns a blank class just to implement struct in python
        """
        return type('', (), {})()
    
    def featureSelectImportance(self, importances, threshold=0.015):
        """function for selecting features above threashold
        """
        # sorting importances
        indices = np.argsort(importances)
        
        data = {}
        # importance unsorted
        data['importances'] = importances   
        # sorted indices according to importance
        data['sortedIndices'] = indices    
        # sorted feature names according to importances
        data['sortedFeatureNames'] = [self.featureNames[x] for x in indices]
        # selected indices above threshold
        selectedIndices = [index for index, value in enumerate(data['importances']) if value > threshold] 
        data['selectedIndices'] = selectedIndices
        data['threshold'] = threshold
        
        return data
    
    def plotImportances(self, importances, threshold, fileName='sampleImportance', dpi=600):
        """function for plotting sorted importance 
        """
        muliplier = 100
        indices = np.argsort(importances)
        L = importances.size
        # Plot
        plt.figure(figsize=(6,8))
        plt.title('Feature Importances', fontsize = 16)
        ypositions= np.array( range(len(indices)) )
        
        plt.barh(3*ypositions, importances[indices]*muliplier, color='b', align='center')
        x = np.array([threshold, threshold])*muliplier
        y = np.array([-3, 3*L+3])
        # line for threshold
        plt.plot(x, y, color='r', ls='--', lw=2)
        # tick marks
        featureTicks = [self.featureNames[x] for x in indices]
        plt.yticks(3*ypositions, featureTicks, fontsize=10)
        plt.xlabel('Relative Importance (%)', fontsize = 16)
        fileName = fileName + '.png'
        plt.savefig(fileName, dpi=dpi)
        
    def saveDoubleCrossValidData(self, fileName, ValScoreList, ValScoreStdList,
                TestScoreList, TestConfList,
                bestParamList, OuterInnerFoldData, sweepingList,
                OuterFoldNo, InnerFoldNo, scoring, algorithm):
        """saving all necessary data for double cross validation 
        """
        bestParamIndexInSwpList = []
        for param in bestParamList:
            count = 0
            for i in sweepingList:
                if param == i:
                    break
                count +=1
            bestParamIndexInSwpList.append(count)
        data = {}
        data['ValScoreList'] = np.array(ValScoreList)
        data['ValScoreStdList'] = np.array(ValScoreStdList)
        data['TestScoreList'] = np.array(TestScoreList)
        data['TestConfList'] = TestConfList
        data['bestParamList'] = np.array(bestParamList)
        data['OuterInnerFoldData'] = OuterInnerFoldData
        data['sweepingList'] = np.array(sweepingList)
        data['OuterFoldNo'] = OuterFoldNo
        data['InnerFoldNo'] = InnerFoldNo
        data['bestParamIndexInSwpList'] = np.array(bestParamIndexInSwpList)
        data['scoring'] = scoring
        data['algorithm'] = algorithm
        self.saveVariables( data, fileName )
        