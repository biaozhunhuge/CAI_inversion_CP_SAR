import numpy as np
import pandas as pd
import datetime
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import time
from sklearn import preprocessing
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import Ridge, LassoCV,LassoLarsCV, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from scipy.stats import skew
from IPython import embed

def create_submission(prediction,score):
    now = datetime.datetime.now()
    sub_file = 'submission_'+str(score)+'_'+str(now.strftime("%Y-%m-%d-%H-%M"))+'.csv'
    #sub_file = 'prediction_training.csv'
    print ('Creating submission: ', sub_file)
    pd.DataFrame({'Id': test['Id'].values, 'SalePrice': prediction}).to_csv(sub_file, index=False)

# train need to be test when do test prediction
def data_preprocess(train):
    #outlier_idx = [4,11,13,20,46,66,70,167,178,185,199, 224,261, 309,313,318, 349,412,423,440,454,477,478, 523,540, 581,588,595,654,688, 691, 774, 798, 875, 898,926,970,987,1027,1109, 1169,1182,1239, 1256,1298,1324,1353,1359,1405,1442,1447]
    #train.drop(train.index[outlier_idx],inplace=True)
    #all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
    #                      test.loc[:,'MSSubClass':'SaleCondition']))
    all_data = train.loc[:,'RSoV':'Raney_Odd']    
    all_data = all_data.values

    #to_delete = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']
    #all_data = all_data.drop(to_delete,axis=1)
    #
    #train["SalePrice"] = np.log1p(train["SalePrice"])
    ##log transform skewed numeric features
    #numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
    #skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
    #skewed_feats = skewed_feats[skewed_feats > 0.75]
    #skewed_feats = skewed_feats.index
    #all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
    #all_data = pd.get_dummies(all_data)
    #all_data = all_data.fillna(all_data.mean())
    #X_train = all_data[:train.shape[0]]
    #X_test = all_data[train.shape[0]:]
    y = train.LAI.values
    _num = y.shape[0]
    test_num = int(_num * 0.3)
    test_idx = np.random.choice(_num, test_num, replace = False)
    train_idx = np.setdiff1d(range(_num), test_idx, assume_unique = True)

    return all_data[train_idx].copy(),y[train_idx].copy(),all_data[test_idx].copy(),y[test_idx].copy()


def mean_squared_error_(ground_truth, predictions):
    return mean_squared_error(ground_truth, predictions) ** 0.5
RMSE = make_scorer(mean_squared_error_, greater_is_better=False)

class ensemble(object):
    def __init__(self, n_folds, stacker, base_models):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
    def fit_predict(self,train,test,ytr):
        X = train
        y = ytr
        T = test
        #folds = list(KFold(len(y), n_splits = self.n_folds, shuffle = True, random_state = 0))
        folds = KFold(self.n_folds,shuffle=True)
        S_train = np.zeros((X.shape[0],len(self.base_models)))
        S_test = np.zeros((T.shape[0],len(self.base_models))) 
        for i,reg in enumerate(base_models):
            print ("Fitting the base model...")
            S_test_i = np.zeros((T.shape[0],self.n_folds)) 
            #for j, (train_idx,test_idx) in enumerate(folds):
            j = 0
            for train_idx, test_idx in folds.split(X):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                reg.fit(X_train,y_train)
                y_pred = reg.predict(X_holdout)[:]
                S_train[test_idx,i] = y_pred
                S_test_i[:,j] = reg.predict(T)[:]
                j += 1
            S_test[:,i] = S_test_i.mean(1)
         
        print ("Stacking base models...")
        # tuning the stacker
        param_grid = {
             'alpha': [1e-3,5e-3,1e-2,5e-2,1e-1,0.2,0.3,0.4,0.5,0.8,1e0,3,5,7,1e1],
        }
        grid = GridSearchCV(estimator=self.stacker, param_grid=param_grid, n_jobs=1, cv=5, scoring=RMSE)
        grid.fit(S_train, y)
        try:
            print('Param grid:')
            print(param_grid)
            print('Best Params:')
            print(grid.best_params_)
            print('Best CV Score:')
            print(-grid.best_score_)
            print('Best estimator:')
            print(grid.best_estimator_)
            print(message)
        except:
            pass

        y_pred = grid.predict(S_test)[:]
        return y_pred, -grid.best_score_

# read data, build model and do prediction
root = r'C:\workspace\wulab\pla_regression'
train = pd.read_excel('%s/data/%03d.xlsx'%(root, 0)) # read train data

# build a model library (can be improved)
base_models = [
        RandomForestRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features=16
        ),
        RandomForestRegressor(
            n_jobs=1, random_state=0,
            n_estimators=500, max_features=16,
	    max_depth = 7
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=0, 
            n_estimators=500, max_features=15
        ),
        ExtraTreesRegressor(
            n_jobs=1, random_state=0, 
          n_estimators=500, max_features=16
        ),
        GradientBoostingRegressor(
            random_state=0, 
            n_estimators=500, max_features=10, max_depth=6,
            learning_rate=0.05, subsample=0.8
        ),
	GradientBoostingRegressor(
            random_state=0, 
            n_estimators=500, max_features=15, max_depth=6,
            learning_rate=0.05, subsample=0.8
        ),
        XGBRegressor(
            seed=0,
            n_estimators=500, max_depth=10,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.75
        ),
 
        XGBRegressor(
            seed=0,
            n_estimators=500, max_depth=7,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.75
        ),
	LassoCV(alphas = [1, 0.1, 0.001, 0.0005]),
	KNeighborsRegressor(n_neighbors = 5),
       	KNeighborsRegressor(n_neighbors = 10),
      	KNeighborsRegressor(n_neighbors = 15),
        KNeighborsRegressor(n_neighbors = 25),
	LassoLarsCV(),
	ElasticNet(),
	SVR()
    ]

ensem = ensemble(
        n_folds=5,
	stacker=Ridge(),
        base_models=base_models
    )

#X_train,X_test,y_train = data_preprocess(train)
X_train,y_train,X_test,ytest = data_preprocess(train)
y_pred, score = ensem.fit_predict(X_train,X_test,y_train)

_mae = np.abs(y_pred - ytest).mean()
print ('mae')
print (_mae)
#create_submission(np.expm1(y_pred),score)


