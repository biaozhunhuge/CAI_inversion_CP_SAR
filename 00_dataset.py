#!coding=utf-8
import numpy as np
import pandas
from sklearn import preprocessing
from IPython import embed

def parse_xlsx(pth):
    df = pandas.read_excel(pth)
    data_set = df.iloc[:, 2:].values.tolist()
    data_set = np.array(data_set)
    return data_set

if __name__ == '__main__':
    root = r'C:\workspace\wulab\pla_regression_v9'
    if 1:
        for e in range(1):
            e = 'PAI'
            xls_pth = '%s/data/%s.xlsx'%(root, e)
            _data = parse_xlsx(xls_pth)

            
            X = _data[:, 1:].copy()
            Y = _data[:, 0].copy()

            x_stand_scaler = preprocessing.StandardScaler()
            X_scaler = x_stand_scaler.fit_transform(X)

            y_stand_scaler = preprocessing.StandardScaler()
            Y_scaler = y_stand_scaler.fit_transform(Y.reshape(-1, 1))
            Y_scaler = Y_scaler.flatten()

            _num = X_scaler.shape[0]
            test_num = int(_num * 0.3)
            test_idx = np.random.choice(_num, test_num, replace = False)
            train_idx = np.setdiff1d(range(_num), test_idx, assume_unique = True)
            X_train = X_scaler[train_idx]
            Y_train = Y_scaler[train_idx]
            X_test = X_scaler[test_idx]
            Y_test = Y_scaler[test_idx]

            np.savez('%s/data/%s.npz'%(root, e),
                    **{
                        'X': X,
                        'Y': Y,
                        'X_norm': X_scaler,
                        'Y_norm': Y_scaler,
                        'X_mean': x_stand_scaler.mean_,
                        'X_std':  x_stand_scaler.scale_,
                        'Y_mean': y_stand_scaler.mean_,
                        'Y_std':  y_stand_scaler.scale_,
                        'X_train': X_train,
                        'X_test': X_test,
                        'Y_train': Y_train,
                        'Y_test': Y_test,
                        'train_idx': train_idx,
                        'test_idx':  test_idx,
                        })