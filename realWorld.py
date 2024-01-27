import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from biaswrappers.regressor import BiasRegressorC2
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import copy

# data from kaggle
# https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

label = "LandAverageTemperature"

df = pd.read_csv("climateChangeData\\GlobalTemperatures.csv")
df = df[[label]]

mergeDf = pd.DataFrame(columns=df.columns)
i = 0
entry = 0
for idx, row in df.iterrows():
    i += 1
    i %= 12
    if i == 0:
        mergeDf.loc[len(mergeDf.index)] = [row[0]]
        entry = 0

df = mergeDf
df = df.reset_index().rename(columns={"index":"date"})

df.dropna(inplace=True)

for lag in [0, 2, 5, 20]:

    # Add previous n-values
    for i in range(lag):
        
        df[label + f'_{i+1}'] = df[label].shift(i+1)
        
        # For simplicity we drop the null values 
        df.dropna(inplace=True)

    innerModel = LinearRegression()
    bc2 = BiasRegressorC2(model=copy.deepcopy(innerModel), postModel=copy.deepcopy(innerModel), split_size=0.3)

    X = df.drop(label, axis=1).values
    y = df[label].values.reshape(-1, 1)

    trainsize = 0.7

    X_train = X[:int(len(X) * trainsize)]
    X_test = X[int(len(X) * trainsize):]

    y_train = y[:int(len(y) * trainsize)]
    y_test = y[int(len(y) * trainsize):]

    innerModel.fit(X_train, y_train)
    bc2.fit(X_train, y_train)

    rawPreds = innerModel.predict(X_test)
    print(f"Lag of {lag}:")
    print("Raw MSE: ")
    print(mean_squared_error(y_test, rawPreds))

    bc2Preds = bc2.predict(X_test)
    print("BC2 MSE: ")
    print(mean_squared_error(y_test, bc2Preds))
    print()

    plt.scatter(X_train[:,0], y_train, color='b')
    plt.scatter(X_test[:,0], y_test, color='tab:orange')
    plt.plot(X_test[:,0], rawPreds, color='r')
    plt.plot(X_test[:,0], bc2Preds, color='g')
    plt.show()
