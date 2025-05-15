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
        e = 0
        print (e)
        buf = np.load('%s/data/%03d.npz'%(root, e))
        XTrain = buf['X_train']
        YTrain = buf['Y_train']
        XTest = buf['X_test']
        YTest = buf['Y_test']
        print (XTrain.shape)

        model = RandomForestRegressor(random_state=137)
        model = KFoldTrian(model, XTrain, YTrain, nFold)
        err = Test(model, XTest, YTest)
        for e in range(4):
            print (err[e])
        print ('\n')

        feat_scores = model.feature_importances_.copy()
        _sort    = (np.argsort(feat_scores))[::-1]


        test_error = float('+inf')
        test_metric = None
        best_model = None
        feat_num = 0
        for keep_num in range(1, 17):
            print (keep_num)
            _keep = _sort[:keep_num]     
            model = RandomForestRegressor(random_state=137)      
            model = KFoldTrian(model, XTrain[:,_keep].copy(), YTrain, nFold)
            err = Test(model, XTest[:,_keep].copy(), YTest)

            if err[1] < test_error:
                test_error = err[1]
                test_metric = err
                best_model = model
                feat_num = keep_num

        print (feat_num)
        for i in range(4):
            print (test_metric[i])
        embed()

if __name__ == '__main__':
    main()