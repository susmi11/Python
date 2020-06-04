# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 19:53:41 2019

@author: Susmita Mathew

Problem Statements

4.1 Import the csv file of the stock which contained the Bollinger columns as well.
Create a new column 'Call' , whose entries are -
'Buy' if the stock price is below the lower Bollinger band
'Hold Buy/ Liquidate Short' if the stock price is between the lower and middle Bollinger band
'Hold Short/ Liquidate Buy' if the stock price is between the middle and upper Bollinger band
'Short' if the stock price is above the upper Bollinger band
Now train a classification model with the 3 bollinger columns and the stock price as inputs and 'Calls' as output. Check the accuracy on a test set. (There are many classifier models to choose from, try each one out and compare the accuracy for each)
Import another stock data and create the bollinger columns. Using the already defined model, predict the daily calls for this new stock.

4.2 Now, we'll again utilize classification to make a trade call, and measure the efficiency of our trading algorithm over the past two years. For this assignment , we will use RandomForest classifier.
Import the stock data file of your choice
Define 4 new columns , whose values are:
% change between Open and Close price for the day
% change between Low and High price for the day
5 day rolling mean of the day to day % change in Close Price
5 day rolling std of the day to day % change in Close Price
Create a new column 'Action' whose values are:
1 if next day's price(Close) is greater than present day's.
(-1) if next day's price(Close) is less than present day's.
i.e. Action [ i ] = 1 if Close[ i+1 ] > Close[ i ]
i.e. Action [ i ] = (-1) if Close[ i+1 ] < Close[ i ]
Construct a classification model with the 4 new inputs and 'Action' as target
Check the accuracy of this model , also , plot the net cumulative returns (in %) if we were to follow this algorithmic model

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import utils
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
sns.set_style('whitegrid')

##4.1
df1 = pd.read_csv('D:\susmi\stocks\week3.csv')
df1['Call'] = 0
for p in range(13,494):
    lb = df1.at[p,'Lower band']
    mb = df1.at[p,'average']
    ub = df1.at[p,'Upper band']
    n = df1.at[p,'Close Price']
    if n<lb:
        df1.loc[p,'Call'] = "Buy"
    elif n>lb and n<mb:
        df1.loc[p,'Call'] = "Hold Buy/ Liquidate Short"
    elif n>mb and n<ub:
        df1.loc[p,'Call'] = "Hold Short/ Liquidate Buy"
    elif n>ub:
        df1.loc[p,'Call'] = "Short"
print(df1['Call'].tail())
#using k nearest neighbor
X = df1.iloc[13:494,[25,28,27,10]].values
Y = df1.iloc[13:494,29].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors = 15)
knn.fit(x_train, y_train) 
y_pred = knn.predict(x_test)
#checking accuracy
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
#using Stochastic Gradient Descent
sgd = SGDClassifier(loss='modified_huber',shuffle=True,random_state = 101)
sgd.fit(x_train, y_train) 
y_pred = sgd.predict(x_test)
print(accuracy_score(y_test, y_pred))
#using naive-bayes
nb = GaussianNB()
nb.fit(x_train, y_train) 
y_pred = nb.predict(x_test)
print(accuracy_score(y_test, y_pred))
#using SVM
svm = SVC(kernel = "linear", C = 0.025, random_state = 101)
svm.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(accuracy_score(y_test, y_pred))

#predict for a different stock
df2 = pd.read_csv('D:\susmi\stocks\BERGEPAINT.csv')
#bollinger bands
df2['average'] = df2['Close Price'].rolling(14, win_type=None).mean()
df2['std'] = df2['Close Price'].rolling(14, win_type=None).std()
df2['Upper band'] = df2['average'] + 2*df2['std']
df2['Lower band'] = df2['average'] - 2*df2['std']
print(df2.head())
#plt.plot(df2['Date'], df2['Upper band'], df2['Date'], df2['average'], df2['Date'], df2['Lower band'], df2['Date'], df2['Average Price'])
#plt.gca().legend(('Upper band', 'Average', 'Lower band', 'Daily Average'))

#using defined model
df2_x_test = df2.iloc[13:494,[15,18,17,8]].values
y_pred = knn.predict(df2_x_test)

##4.2
df3 = pd.read_csv('D:\susmi\stocks\CASTROLIND.csv')
df3['pct_change_open_close'] = ((df3['Close Price']-df3['Open Price'])/df3['Close Price'])*100
df3['pct_change_low_high'] = ((df3['High Price']-df3['Low Price'])/df3['High Price'])*100
df3['Day_Perc_Change'] = df3['Close Price'].pct_change()
df3.loc[0,'Day_Perc_Change'] = 0
df3['closing mean'] = df3['Day_Perc_Change'].rolling(5, win_type=None).mean()
df3['closing std'] = df3['Day_Perc_Change'].rolling(5, win_type=None).std()
df3['Action'] = 0
for i in range(0,494):
    if (df3.iloc[i+1, 8] > df3.iloc[i, 8]):
        df3.iloc[i,20] = 1
    else:
        df3.iloc[i,20] = -1
X = df3.iloc[4:494,[15,16,18,19]].values
Y = df1.iloc[4:494,20].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
rfm = RandomForestClassifier(n_estimators = 70, min_samples_leaf = 30, n_jobs = -1, oob_score = True, random_state = 101)
#encoding because of 'ValueError: Classification metrics can't handle a mix of continuous and multiclass targets'
lab_enc = preprocessing.LabelEncoder()
y_training_scores_encoded = lab_enc.fit_transform(y_train)
print(y_training_scores_encoded)
print(utils.multiclass.type_of_target(y_train))
print(utils.multiclass.type_of_target(y_train.astype('int')))
print(utils.multiclass.type_of_target(y_training_scores_encoded))

y_testing_scores_enc = lab_enc.fit_transform(y_test)
print(y_testing_scores_enc)
print(utils.multiclass.type_of_target(y_test))
print(utils.multiclass.type_of_target(y_test.astype('int')))
print(utils.multiclass.type_of_target(y_testing_scores_enc))

rfm.fit(x_train, y_training_scores_encoded)
y_pred = rfm.predict(x_test)
print(accuracy_score(y_testing_scores_enc, y_pred))

#cumulative net returns
df3['Cumulative net'] = df3['pct_change_open_close'].cumsum()
plt.plot(df3['Cumulative net'])
