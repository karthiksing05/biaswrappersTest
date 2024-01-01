# models
from biaswrappers.regressor import BiasRegressorC2

# tested models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# sklearn free datasets
from sklearn.datasets import load_diabetes, load_linnerud, fetch_california_housing, make_regression
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3

# metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import copy
import os

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=RuntimeWarning)
simplefilter("ignore", category=ConvergenceWarning)

def get_model_name(model):
    return str(model.__class__).split('.')[-1][:-2]

old_school = [
    LinearRegression()
]
penalized_lr = [Lasso(tol=0.002), Ridge(), ElasticNet()]
dtrees = [DecisionTreeRegressor(max_depth=md) for md in [5, 10]]
rfrs = [RandomForestRegressor(n_estimators=n) for n in [5, 10]]

reg_models = old_school + penalized_lr + dtrees + rfrs

def load_ransac():
    X, y, _ = make_regression(
        n_samples=1000,
        n_features=1,
        n_informative=1,
        noise=10,
        coef=True,
        random_state=0,
    )

    # Add outlier data
    np.random.seed(0)
    X[:50] = 3 + 0.5 * np.random.normal(size=(50, 1))
    y[:50] = -3 + 10 * np.random.normal(size=50)
    return X, y

datasets = {
    "Diabetes": load_diabetes(return_X_y=True), # default test
    "Linnerud": load_linnerud(return_X_y=True), # multioutput
    "RANSAC_Data": load_ransac(), # outlier test
    "Friedman_1": make_friedman1(noise=5), # useless features
    "Friedman_2": make_friedman2(noise=1), # nonlinear data
    "Friedman_3": make_friedman3(noise=1), # nonlinear data
    "California_Housing": fetch_california_housing(return_X_y=True) # default test
}

def neg_rms_error(estimator, X, y):
    y_pred = estimator.predict(X)
    while len(list(y_pred.shape)) > len(list(y.shape)):
        y_pred = y_pred[0]
    mse = mean_squared_error(y, y_pred)
    return -1 * np.sqrt(mse)

bestInnerModel = {}
bestPostModel = {}
bestModels = {}

for name, dataset in datasets.items():
    X, y = dataset
    cv = KFold(n_splits=10, shuffle=True)

    scores = {}
    for xmodel in reg_models:
        for postModel in reg_models:
            model = BiasRegressorC2(model=copy.deepcopy(xmodel), postModel=copy.deepcopy(postModel))
            cvScores = cross_val_score(model, X, y, scoring=neg_rms_error, cv=cv, n_jobs=1)

            inner = (
                get_model_name(model.model) + 
                str(model.model.get_params().get('max_depth', "")) + 
                str(model.model.get_params().get('n_neighbors', "")) + 
                str(model.model.get_params().get('n_estimators', ""))
            )

            post = (
                get_model_name(model.postModel) + 
                str(model.postModel.get_params().get('max_depth', "")) + 
                str(model.postModel.get_params().get('n_neighbors', "")) + 
                str(model.postModel.get_params().get('n_estimators', ""))
            )

            key = get_model_name(model)
            key += " with inner model " + inner
            key += " and with post model " + post

            scores[key] = [np.mean(cvScores), np.std(cvScores), model]
            if inner not in bestInnerModel:
                bestInnerModel[inner] = 0
            bestInnerModel[inner] += np.mean(cvScores)
            if post not in bestPostModel:
                bestPostModel[post] = 0
            bestPostModel[post] += np.mean(cvScores)
            if key not in bestModels:
                bestModels[key] = 0
            bestModels[key] += np.mean(cvScores)

    modeldf = pd.DataFrame.from_dict(scores, orient='index').sort_values(0, ascending=False)
    modeldf.columns = ['RMSE_MEAN', 'RMSE_STD', 'MODEL_CLASS']

    dfd = modeldf.to_dict()
    best_model = str(max(dfd["RMSE_MEAN"], key=dfd["RMSE_MEAN"].get))
    model = dfd['MODEL_CLASS'][best_model]
    print("Best Model for {}: {}\n".format(name, best_model))
    print("Top 5 models, sorted by RMSE:")
    print(modeldf.head())
    print("-----------------------------------------------")
    modeldf.to_csv(f"resC2\\{name}.csv")

print("Saved all to CSV! @ resC2")
bestInnerModel = dict(sorted(bestInnerModel.items(), key=lambda x: x[1], reverse=True))
bestPostModel = dict(sorted(bestPostModel.items(), key=lambda x: x[1], reverse=True))
bestModels = dict(sorted(bestModels.items(), key=lambda x: x[1], reverse=True))

bestInnerDf = pd.DataFrame.from_dict(bestInnerModel, orient='index')
bestInnerDf.columns = ["RMSE_SUM"]
bestInnerDf.to_csv("resC2\\c2InnerRes.csv")

bestPostDf = pd.DataFrame.from_dict(bestPostModel, orient='index')
bestPostDf.columns = ["RMSE_SUM"]
bestPostDf.to_csv("resC2\\c2PostRes.csv")

bestDf = pd.DataFrame.from_dict(bestModels, orient="index")
bestDf.columns = ['RMSE_SUM']
bestDf.to_csv("resC2\\c2Results.csv")