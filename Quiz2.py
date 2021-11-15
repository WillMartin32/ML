import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression

train = pd.read_csv("gameratings.csv")
test = pd.read_csv("test_esrb.csv")

#34 columns

x_train = train.values.reshape(-1,33)

y_train = train.target.values.reshape(-1,1)


knn = KNeighborsClassifier()

knn.fit(X=x_train ,y=y_train)

print('Done')


