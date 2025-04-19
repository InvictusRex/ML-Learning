import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# One Hot Encoding the State column
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# We don't need to worry about Dummy Variable Trap & don't have to manually remove one of the columns
# The class for Linear Regression in sklearn will take care of that for us
# Similarly, we don't need to specailly define what type of model to build like Backward Elimination
# The class for Linear Regression in sklearn will automatically choose the best features & build it accordingly

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# For displaying, we can't display all the 4 independant variables in a 2D graph, along with predictd profit, it will need a 5D graph
# So we will display the predicted profit vs the actual profit for the test set

y_pred = regressor.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))

# For predicting for a specific value
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Getting the linear regression equation & values of coefficients
print(regressor.coef_)
print(regressor.intercept_)