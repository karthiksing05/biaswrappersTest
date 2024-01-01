# models
from biaswrappers.regressor import BiasRegressorC1, BiasRegressorC2, RandomWrapper

# tested models
from sklearn.linear_model import LinearRegression, Lasso, Ridge

# sklearn free datasets
from sklearn.datasets import make_regression

# metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

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
penalized_lr = [Lasso(tol=0.002), Ridge()]
my_penalized_lr = [BiasRegressorC1(split_size=0.65), BiasRegressorC2(split_size=0.1), RandomWrapper()]

reg_models = old_school + penalized_lr + my_penalized_lr

"""
Big Sample, No outliers: n_samples = 1000, noise = 10
Small Sample, No outliers: n_samples = 20, noise = 0.2
Big Sample, with outliers: n_samples = 1000, noise = 10, uncomment the two lines below make_regression
"""

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

def rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

plt.scatter(X_train, y_train, color='k')

colors = ['r', 'y', 'g', 'c', 'm', 'b']

scores = {}
for i, model in enumerate(reg_models):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if list(y_pred.shape)[0] == 1:
        y_pred = np.array([np.array(x) for x in y_pred[0]])
    
    key = get_model_name(model)

    plt.plot(X_test, y_pred, color=colors[i], label=key)

    scores[key] = rmse(y_test, y_pred)

print(scores)
leg = plt.legend(loc='upper center')
plt.show()