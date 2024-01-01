# models
from biaswrappers.regressor import BiasRegressorC1, BiasRegressorC2, RandomWrapper

# tested models
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)

# sklearn free datasets
from sklearn.datasets import load_diabetes, load_linnerud, fetch_california_housing, make_regression
from sklearn.datasets import make_friedman1, make_friedman2, make_friedman3

# UCI datasets
from ucimlrepo import fetch_ucirepo

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

np.random.seed(42)

def get_model_name(model):
    return str(model.__class__).split('.')[-1][:-2]

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
    "Friedman_2": make_friedman2(noise=1),
    "Friedman_3": make_friedman3(noise=1),
    "California_Housing": fetch_california_housing(return_X_y=True) # default test
}

def neg_rms_error(estimator, X, y):
    y_pred = estimator.predict(X)
    while len(list(y_pred.shape)) > len(list(y.shape)):
        y_pred = y_pred[0]
    mse = mean_squared_error(y, y_pred)
    return -1 * np.sqrt(mse)

rankingsC1 = dict([(round(x, 2), 0) for x in list(np.arange(0.1, 1.0, 0.05))])
rankingsC2 = dict([(round(x, 2), 0) for x in list(np.arange(0.1, 1.0, 0.05))])

for name, dataset in datasets.items():
    X, y = dataset
    cv = KFold(n_splits=10, shuffle=True)

    scores = {}
    for split_size in list(np.arange(0.1, 1.0, 0.05)):
        split_size = round(split_size, 2)
        models = [BiasRegressorC1(split_size=split_size), BiasRegressorC2(split_size=split_size)]
        cvScoresC1 = cross_val_score(models[0], X, y, scoring=neg_rms_error, cv=cv, n_jobs=1)
        cvScoresC2 = cross_val_score(models[1], X, y, scoring=neg_rms_error, cv=cv, n_jobs=1)
        key = split_size

        rankingsC1[key] += np.mean(cvScoresC1)
        rankingsC2[key] += np.mean(cvScoresC2)

print(dict(sorted(rankingsC1.items(), key=lambda x: x[1], reverse=True)))
print(dict(sorted(rankingsC2.items(), key=lambda x: x[1], reverse=True)))