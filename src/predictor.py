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
import matplotlib.pyplot as plt
#from math import pi, sqrt, sin, cos, tan
#from sets import Set
import pickle
import sys
import os
import shutil
import operator as op
import time
import argparse
import scipy.io as sio
import logging
import seaborn as sn

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
        
        self.model = []
        
        self.fTrain = []
        self.fTest = []
        self.cTrain = []
        self.cTest = []
        self.baseLogger = []
        
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
                 feaRowEnd = 284808, delimiter = ','):
        """loading data
           @param fileName - name of the csv file
           @param fearowStart - starting row of the feature or Class
           @param feaRowEnd - end row of the feature or Class
           @param colClass - number of the col for Class (indexing starts from 0)
           @param delimitter - default comma
        """
        self.dF = pd.read_csv( fileName, delimiter = delimiter)
        #if feaRowStart is not None and feaRowEnd is not None \
               #colFeaStart is not None and colFeaEnd is not None :
        self.feature = self.dF.iloc[feaRowStart-1:feaRowEnd, 
                                    colFeaStart-1:colFeaEnd]
        self.Class = self.dF.iloc[feaRowStart-1:feaRowEnd, colClass-1]
        self.dataNo = feaRowEnd - feaRowStart + 1
        self.featureNo = colFeaEnd - colFeaStart + 1
        print('Database loaded with: \n \
                  \t Feature No - %d \n \
                  \t Total data points - %d\n' % (self.featureNo, self.dataNo))
        #else:
        #    self.feature = self.dF.iloc[:,0:30]
        #    self.Class = self.dF.Class 
            #self.feature = self.dF.iloc[:, 0:self.dF.shape[1]-1]
            #self.Class = self.dF.loc[:, self.dF.shape[1]-1:self.dF.shape[1]]  
            
        self.dataConvertToNumpy()
        
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
    
    def getMetrics( self, classTest, classPred, boolPrint = True):
        """copying unbalanced for multiple times to balance dataset
           @param classTest
           @param classPred
           @return accruacy - one number
           @return matConf - 2x2 matrix for 2 class classifier
           @return matCohenKappa - TODO
           @return strClassificationReport - TODO
        """
        # accuracy of the model - in one number
        accuracy = average_precision_score( classTest, classPred )
        # confusion matrix 2x2 matric
        matConf = confusion_matrix(classTest, classPred)
        # cohen Kappa is applicable for unbalanced data
        matCohenKappa = cohen_kappa_score(classTest, classPred)
        # classification report
        strClassificationReport = classification_report(classTest, classPred)
        
        if boolPrint is True:
            print('Avg. Precision Score = %0.2f %%\n' % (accuracy*100) )
            print('Classification Report:\n = %s\n' % (strClassificationReport) )
            print('Confusion Matrix:\n' )
            print(matConf)
            print('\n')
        
        return accuracy, matConf, matCohenKappa, strClassificationReport
    
    def trainModel( self, featureTrain, classTrain, pModel=None):
        """overriding virtual function for training
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel
        # training the model
        print('Traning model ...')
        self.model.fit(featureTrain, classTrain)
        return self.model
    
    def testModel( self, featureTest, pModel=None):
        """overriding virtual function for testing
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel 
        print('Traning model ...')
        # predicting with trained model
        classPred = self.model.predict( featureTest )
    
        return classPred

#    def trainModel( self, featureTrain, classTrain):
#        """virtual function for training
#        """
#        print('Traning model ...')
#        
#    def testModel( self, classTest, classPred):
#        """virtual function for testing
#        """
#        print('Testing model ...')
    
    def printConfusionMatrix( self, confMatrix, 
                             classLabels=["0", "1"],
                             filename='confusionMatrix', dpi=600, cmap=None ):
        
        cmap = cmap or sn.cubehelix_palette(dark=0, light=1, as_cmap=True)
        df_cm = pd.DataFrame(confMatrix, index = [i for i in classLabels],
                  columns = [i for i in classLabels])
        plt.figure(figsize = (2.5,2))
        sn.heatmap(df_cm, annot=True, cmap=cmap)
        plt.savefig( filename+".png", dpi=dpi )
        # normalizing matrix
#        norm_conf = []
#        for i in confMatrix:
#            a = 0
#            tmp_arr = []
#            a = sum(i, 0)
#            for j in i:
#                tmp_arr.append(float(j)/float(a))
#            norm_conf.append(tmp_arr)
#        
#        fig = plt.figure()
#        plt.clf()
#        ax = fig.add_subplot(111)
#        ax.set_aspect(1)
#        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
#                        interpolation='nearest')
#        
#        width, height = confMatrix.shape
#        
#        for x in xrange(width):
#            for y in xrange(height):
#                ax.annotate(str(confMatrix[x][y]), xy=(y, x), 
#                            horizontalalignment='center',
#                            verticalalignment='center')
#        
#        cb = fig.colorbar(res)
#        #alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#        plt.xticks(range(classLabels), classLabels[:width])
#        plt.yticks(range(classLabels), classLabels[:height])
#        plt.savefig(filename+".png", dpi=dpi)
        
    def crossValidate(self, pfeatures, pClass, nFold=5, pModel=None, 
                      scoring='accuracy'):
        """function for cross validation
        """
        # if model is given, override with internal model
        if pModel is not None:
            self.model = pModel 
        scores = cross_val_score(self.model, pfeatures, pClass, cv=nFold, 
                                 scoring=scoring)
        return scores, scores.mean(), scores.std()