import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the Linnerud dataset
linnerud = datasets.load_linnerud()
X, y = linnerud.data, linnerud.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a DataFrame for the Linnerud dataset
df = pd.DataFrame(data=X, columns=linnerud.feature_names)
yDf = pd.DataFrame(data=y, columns=linnerud.target_names)
df = pd.concat([df, yDf], axis=1)
# Calculate the correlation matrix
print(df.head())
correlation_matrix = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(4, 3))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()