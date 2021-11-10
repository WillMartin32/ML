''' Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line '''

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import seaborn as sns

diabetes = load_diabetes()

#how many sameples and How many features?
#print(diabetes.data.shape)


# What does feature s6 represent?
#print(diabetes.DESCR)
# glu, blood sugar level



#print out the coefficient
x_train, x_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)


# There are three steps to model something with sklearn
# 1. Set up the model
lr = LinearRegression()

# 2. Use the fit to train our model
lr.fit(X=x_train, y=y_train)

# print out the coefficint
coef = lr.coef_
print(coef)


#print out the intercept
intercept = lr.intercept_
print(intercept)

# 3. Use predict to test your model
predicted = lr.predict(x_test)
expected =  y_test

# create a scatterplot with regression line

plt.plot(expected, predicted, ".")



x = np.linspace(0,330,100)
print(x)
y = x

plt.plot(x,y)
plt.show()