import numpy as np
import pandas as pd
from sklearn.metrics import recall_score , average_precision_score
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

credData = []
classData = []

def readData(csvFile):
    df = pd.read_csv(csvFile, delimiter = ',')
    global credData, classData
    credData = df.iloc[:,0:df.shape[1]-1]
    classData = df.iloc[:,df.shape[1]-1:df.shape[1]]
    credData = np.asarray(credData)
    classData = np.asarray(classData)


def trainModel():
    global credData, classData
    print credData
    X_train, X_test, y_train, y_test = train_test_split(credData, classData, test_size = 0.5, random_state=42)
    d_train =  xgb.DMatrix(X_train, y_train)
    d_test = xgb.DMatrix(X_test, y_test)
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    param['nthread'] = 4
    param['eval_metric'] = 'auc'
    plst = param.items()
    num_round = 10
    evallist = [(d_test, 'eval'), (d_train, 'train')]
    bst = xgb.train(param, d_train, num_round, evallist)
    preds = bst.predict(d_test)
    labels = d_test.get_label()
    print ('error=%f' % ( sum(1 for i in range(len(preds)) if int(preds[i]>0.5)!=labels[i]) /float(len(preds))))
    bst.save_model('0001.model')
    # dump model
    bst.dump_model('dump.raw.txt')
    # dump model with feature map
    #bst.dump_model('dump.nice.txt','featmap.txt')

def main():
    csvFile = 'creditcard.csv'
    readData(csvFile)

    trainModel()

if __name__ == '__main__':
    main()
