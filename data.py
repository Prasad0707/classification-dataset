# -*- coding: utf-8 -*-
"""
Created on Mon May 18 17:00:41 2020

@author: Duddupudi
"""
import pandas as pd


df=pd.read_csv(r"C:\Users\Duddupudi\Downloads\data.csv")
df['timestamp']=pd.to_datetime(df.timestamp,errors='coerce')


df['year'] = df['timestamp'].dt.year 
df['month'] = df['timestamp'].dt.month 
df['day'] = df['timestamp'].dt.day 
df['hour'] = df['timestamp'].dt.hour 
df['minute'] = df['timestamp'].dt.minute

df=df.dropna(axis=0)
df
df.dtypes
x=df.iloc[:,[7,8,9,10,11,12,13]].values
y=df.iloc[:,[1,2,3,4,5,6]].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=9)




from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=1)
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
from sklearn import metrics
print(metrics.accuracy_score(y_test,y_pred)*100)

from sklearn.tree import DecisionTreeClassifier
modeldt=DecisionTreeClassifier(criterion='entropy')#ginni
modeldt.fit(x_train,y_train)
y_preddt=modeldt.predict(x_test)
from sklearn import metrics
print('Decisiontree',metrics.accuracy_score(y_test,y_preddt)*100)
