import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

test = pd.read_csv("gameratings.csv")
target = pd.read_csv("test_esrb.csv")


lr = LinearRegression()

lr.fit(X=x_train, y=y_train)

coef = lr.coef_
intercept = lr.intercept_

predicted = lr.predict(x_test)
expected =  y_test


#knn = KNeighborsClassifier()

#knn.fit(X=test ,y=target)




