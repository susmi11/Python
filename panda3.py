# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:51:06 2019

@author: Susmita Mathew
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
sns.set_style('whitegrid')

df = pd.read_csv('GOLD.csv')
#print(df.head())
##3.1
X = df.iloc[0:411,1:5].values
Y = df.iloc[0:411,7].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

cls = linear_model.LinearRegression()
cls.fit(x_train,y_train)
prediction = cls.predict(x_test)

print(cls.get_params())
print('Co-efficient of linear regression',cls.coef_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))

X2 = df.iloc[411:,1:5].values
prediction = cls.predict(X2)
df.iloc[411:,7] = prediction
#print(df.tail())

X = df.iloc[:,1:5].values
Y = df.iloc[:,8].values
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
sec = linear_model.LinearRegression()
sec.fit(x_train,y_train)
prediction = sec.predict(x_test)

print('Co-efficient of linear regression',sec.coef_)
print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))
print("'Pred' column is a linear combination while 'new' column is a polynomial function of ohlc")

plt.plot(prediction)
plt.plot(y_test)

##3.2
df1 = pd.read_csv('APOLLOTYRE.csv')
nifty = pd.read_csv('Nifty50.csv')

print(df1['Date'].dtype)
df1['Date'] = df1['Date'].astype('datetime64[ns]')
print(df1['Date'].dtype)

df1['Daily Return'] = (df1['Close Price']-df1['Open Price'])/df1['Open Price']
nifty['Daily Return'] = (nifty['Close']-nifty['Open'])/nifty['Open']
n = np.size(X)
print(nifty.head())
Y = df1['Daily Return'].values
#Y = X.reshape(-1,1)
X = nifty['Daily Return'].values
#X = Y.reshape(-1,1)
m_x, m_y = X.mean(), Y.mean()
#cross deviation
ss_xy = np.sum(Y*X) - n*m_x*m_y
#deviation about x
ss_xx = np.sum(X*X) - n*m_x*m_y
beta = ss_xy/ss_xx
print("daily beta is: ",beta)

#monthly return
month = pd.DatetimeIndex(df1['Date']).month
df1['Month'] = month
year = pd.DatetimeIndex(df1['Date']).year
df1['Year'] = year
df2 = df1.groupby(['Year','Month'])
#print(df2.first())
month = pd.DatetimeIndex(nifty['Date']).month
nifty['Month'] = month
year = pd.DatetimeIndex(nifty['Date']).year
nifty['Year'] = year
nifty2 = nifty.groupby(['Year','Month'])
#print(nifty2.first())

def monthly_returns(data, year, month,p,q):
    a = data.get_group((year, month))
    #print(a.head())
    l = len(a.index)
    stock_ret = a.iloc[0,p]
    for i in range(l):
        stock_ret = stock_ret*a.iloc[i,q]
    return stock_ret
#ignoring May of 2017 and 2019 as not whole months
listy = []
listx = []
for yr in range(2017,2020):
    for mon in range(1,13):
        if(not((yr == 2017 and mon < 6)or(yr == 2019 and mon > 4))):
            it = monthly_returns(df2,yr,mon,8,15)
            listy.append(it)

for yr in range(2017,2020):
    for mon in range(1,13):
        if(not((yr == 2017 and mon < 6)or(yr == 2019 and mon > 4))):
            it = monthly_returns(nifty2,yr,mon,4,7)
            listx.append(it)
#print(listx)
#print(listy)

m_x = sum(listx)/len(listx)
m_y = sum(listy)/len(listy)
#cross deviation
ss_xy = np.sum(Y*X) - n*m_x*m_y
#deviation about x
ss_xx = np.sum(X*X) - n*m_x*m_y
beta = ss_xy/ss_xx
print("monthly beta is: ",beta)
"""
Value of both monthly and daily beta is greater than one.
This means there is higher risk, but there are also higher returns
The estimation period is short, so the beta indicates the current 
dynamics of the company. Due to this it is also an unreliable beta

"""
