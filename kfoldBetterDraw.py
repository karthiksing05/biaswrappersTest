# models
from biaswrappers.regressor import BiasRegressorC1, BiasRegressorC2, RandomWrapper

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

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=RuntimeWarning)
simplefilter("ignore", category=ConvergenceWarning)

np.random.seed(42)

def get_model_name(model):
    return str(model.__class__).split('.')[-1][:-2]

old_school = [
    LinearRegression()
]
penalized_lr = [Lasso(tol=0.002), Ridge(), ElasticNet()]
dtrees = [DecisionTreeRegressor(max_depth=md) for md in [3, 5, 10]]
rfrs = [RandomForestRegressor(n_estimators=n) for n in [3, 5, 10]]

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

cols = ["Raw Model RMSE", "C1 RMSE", "C2 RMSE", "RandomWrapper RMSE"]

for name, dataset in datasets.items():
    X, y = dataset
    cv = KFold(n_splits=10, shuffle=True)

    scores = {}
    for xmodel in reg_models:
        key = get_model_name(xmodel)
        key += str(xmodel.get_params().get('max_depth', ""))
        key += str(xmodel.get_params().get('n_estimators', ""))
        scores[key] = []
        models = [BiasRegressorC1(model=xmodel, split_size=0.65), BiasRegressorC2(model=xmodel, postModel=copy.deepcopy(xmodel), split_size=0.1), RandomWrapper(model=xmodel), xmodel]
        for i, model in enumerate(models):
            cvScores = cross_val_score(model, X, y, scoring=neg_rms_error, cv=cv, n_jobs=1)

            scores[key].append(-1 * np.mean(cvScores))

    modeldf = pd.DataFrame.from_dict(scores, orient='index').sort_values(0, ascending=False)
    modeldf.columns = cols
    modeldf.to_csv(f"resDraw\\{name}.csv")

print("Saved all to CSV! @ res")