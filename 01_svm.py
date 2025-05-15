#!coding=utf-8
import numpy as np
import pandas
from sklearn.svm import SVR
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import f
from sklearn.metrics import explained_variance_score,mean_absolute_error,mean_squared_error,r2_score
from IPython import embed

def Train(model, X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    model.fit(X,Y)
    return model

def Test(model, X, Y):
    X = np.asarray(X)
    Y = np.asarray(Y)
    y   = model.predict(X)
    ev  = explained_variance_score(Y, y)
    mae = mean_absolute_error(Y, y)
    mse = mean_squared_error(Y, y)
    r2  = r2_score(Y, y)
    meany = Y.mean()
    SSe   = 0
    SSt   = 0
    for i in range(len(y)):
        d = Y[i] - y[i]
        SSe += d * d
        d = Y[i] - meany
        SSt += d * d
    SSr = SSt - SSe
    row, col = X.shape
    DFr = col
    DFe = row - (col + 1)
    DFt = DFr + DFe
    MSe = SSe / DFe
    MSr = SSr / DFr
    MSt = SSt / DFt
    ftest = f.cdf(MSr / MSe, DFr, DFe)
    return [ev, mae, mse, r2,ftest]

def KFoldTrian(model, X, Y, fold ):
    X = np.asarray(X)
    Y = np.asarray(Y)
    kf        = KFold(n_splits=fold,shuffle=False)
    mae       = float('+inf')
    bestModel = None
    for train_index, test_index in kf.split(X):
        X_Train, X_Test = X[train_index], X[test_index]
        Y_Train, Y_Test = Y[train_index], Y[test_index]
        model = Train(model,X_Train,Y_Train)
        err   = Test(model, X_Test, Y_Test)
        if err[1] < mae :
            mae        = err[1]
            bestModel  = model
    return bestModel

def main():
    nFold = 10

    root = r'C:\workspace\wulab\pla_regression'
    for e in range(5):
        e = 4
        print (e)
        buf = np.load('%s/data/%03d.npz'%(root, e))
        X = buf['X']
        Y = buf['Y']
        print (X.shape)
        #X = np.array(  list(range(0,200))  ).reshape(-1, 1)
        #X = np.c_[X, X] * 0.5
        #Y = np.array(  list(range(0,200))  )
        
        x_stand_scaler = preprocessing.StandardScaler()
        X_scaler = x_stand_scaler.fit_transform(X)

        y_stand_scaler = preprocessing.StandardScaler()
        Y_scaler = y_stand_scaler.fit_transform(Y.reshape(-1, 1))
        Y_scaler = Y_scaler.flatten()

        test_error = float('+inf')
        test_metric = None
        best_model = None
        for i in range(100):
            print (i)
            XTrain, XTest, YTrain, YTest = train_test_split(X_scaler, Y_scaler, test_size=0.3)
        
            model = RandomForestRegressor()
            model = KFoldTrian(model, XTrain, YTrain, nFold)
            err = Test(model, XTest, YTest)
            if err[1] < test_error:
                test_error = err[1]
                test_metric = err
                best_model = model

        for i in range(4):
            print (test_metric[i])
        #print(err[0])
        #print(err[1])
        #print(err[2])
        #print(err[3])
        embed()

if __name__ == '__main__':
    main()