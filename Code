import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error

df = pd.read_csv('winequality-red.csv')
df.head()

df.describe()

# There are no categorical variables. Each feature is a number. Regression problem.
# Given the set of values for features, we have to predict the quality of wine.
# Finding correlation of each feature with our target variable - quality.
correlations = df.corr()['quality'].drop('quality') print(correlations)

sns.heatmap(df.corr())
plt.show()

def get_features(correlation_threshold):
abs_corrs = correlations.abs()
high_correlations = abs_corrs[abs_corrs > correlation_threshold].index.values.tol
ist()
return high_correlations

# Taking features with correlation more than 0.05 as input x and quality as target variable y.
features = get_features(0.05)
print(features)
x = df[features]
y = df['quality']

x
y
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)

# x_train.shape
# x_test.shape
# y_train.shape
y_test.shape

# Fitting linear regression to training data
regressor = LinearRegression()
regressor.fit(x_train,y_train)

# This gives the coefficients of the 10 features selected above.
regressor.coef_

train_pred = regressor.predict(x_train)
train_pred

test_pred = regressor.predict(x_test)
test_pred

# calculating rmse
train_rmse = mean_squared_error(train_pred, y_train) ** 0.5
train_rmse

test_rmse = mean_squared_error(test_pred, y_test) ** 0.5
test_rmse

# Rounding off the predicted values for test set.
predicted_data = np.round_(test_pred)
predicted_data

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pre d)))

# Displaying coefficients of each feature.
coeffecients = pd.DataFrame(regressor.coef_,features)
coeffecients.columns = ['Coeffecient']
coeffecients
