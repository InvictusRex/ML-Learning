import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:-1].values 
y = dataset.iloc[:, -1].values

# Basic Linear Regression Model
lin_reg = LinearRegression()
lin_reg.fit(x, y)

# Creating the matrix of features for each of b0x1^1, b1x1^2, ... , bnx1^n
poly_reg = PolynomialFeatures(degree=2) # Here n=2
x_poly = poly_reg.fit_transform(x)

# Polynomial Regression Model
# Combining the matrix of features crated above together with all in a Linear Regression Model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)


#Visualising the Linear Regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Predicted salaries (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color='red')
plt.plot(x, lin_reg_2.predict(x_poly), color='blue')
plt.title('Predicted salaries (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
print(lin_reg.predict([[6.5]]))

# Predicting a new result with Polynomial Regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]])))