import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

data=pd.read_csv('Advertising.csv')

data.describe()

data.info()

#EDA


sns.histplot(x=data.TV,kde=True)
sns.histplot(x=data.Radio,kde=True)
sns.histplot(x=data.Newspaper,kde=True)

sns.relplot(x='TV',y='Sales',data=data)
plt.plot(data.TV)
plt.plot(data.Sales)
sns.relplot(x='Radio',y='Sales',data=data)
sns.relplot(x='Newspaper',y='Sales',data=data)

sns.pairplot(data)


#data preprocesing and feature Engineering


data.isnull().sum()
sns.boxplot(x=data.TV)
sns.boxplot(x=data.Radio)
sns.boxplot(x=data.Newspaper)

l1=['Unnamed: 0']
data.drop(l1,axis=1,inplace=True)

sns.heatmap(data.drop('Sales',axis=1).corr(),annot=True)

#model creation

X=data[['TV','Newspaper','Radio']]
y=data['Sales']
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=100)

y_train

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.predict(X_test)

y_test

y_pred

from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
r2_score(y_test,y_pred)

X_test.shape

import math
print(mean_squared_error(y_test,y_pred))
print(math.sqrt(mean_squared_error(y_test,y_pred)))

print(mean_absolute_error(y_test,y_pred))
