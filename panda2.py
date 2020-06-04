# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:24:38 2019

@author: Susmita Mathew
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

df_csv = pd.read_csv('D:\susmi\stocks\week2.csv')

print(df_csv['Date'].dtype)
df_csv['Date'] = df_csv['Date'].astype('datetime64[ns]')
print(df_csv['Date'].dtype)

plt.plot(df_csv['Date'], df_csv['Close Price'])

##2.2
plt.stem(df_csv['Date'], df_csv['Day_Perc_Change'])

##2.3
plt.plot(df_csv['Date'], df_csv['Total Traded Quantity'])

##2.4

df_csv['Trend'].value_counts().plot(kind='pie')
plt.axis('equal')
plt.title('Number of appearances in dataset')

trend = df_csv.groupby(['Trend']).mean()
print(trend)
plot = trend.plot.bar(y = 'Total Traded Quantity', rot = 0)
trend = df_csv.groupby(['Trend']).median()
print(trend)
plot = trend.plot.bar(y = 'Total Traded Quantity', rot = 0)

##2.5
df_csv['Daily Return'] = (df_csv['Close Price']-df_csv['Open Price'])/df_csv['Open Price']
#print(df_csv.head())
plt.hist(df_csv['Daily Return'], density=True, bins= 100)
plt.xlabel('Daily Returns');

##2.6
df1 = pd.read_csv("D:\susmi\stocks\APOLLOTYRE.csv")
df2 = pd.read_csv("D:\susmi\stocks\BERGEPAINT.csv")
df3 = pd.read_csv("D:\susmi\stocks\CASTROLIND.csv")
df4 = pd.read_csv("D:\susmi\stocks\CUMMINSIND.csv")
df5 = pd.read_csv("D:\susmi\stocks\DHFL.csv")
#removing rows without eq
df1 = df1[df1.Series == 'EQ']
df2 = df2[df2.Series == 'EQ']
df3 = df3[df3.Series == 'EQ']
df4 = df4[df4.Series == 'EQ']
df5 = df5[df5.Series == 'EQ']

"""
#matching indexes to avoid NaN values
df1 = df1.set_index(df_csv.index)        
df2 = df2.set_index(df_csv.index)        
df3 = df3.set_index(df_csv.index)        
df4 = df4.set_index(df_csv.index)        
df5 = df5.set_index(df_csv.index)        
"""
#dataframe with closing prices
freshdf = pd.DataFrame()
freshdf['RELINFRA'] = df_csv['Close Price']
freshdf['APOLLOTYRE'] = df1['Close Price']
freshdf['BERGEPAINT'] = df2['Close Price']
freshdf['CASTROLIND'] = df3['Close Price']
freshdf['CUMMINSIND'] = df4['Close Price']
freshdf['DHFL'] = df5['Close Price']
#dataframe with percentage change
fresh = pd.DataFrame()
fresh['RELINFRA'] = freshdf['RELINFRA'].pct_change()
fresh['APOLLOTYRE'] = freshdf['APOLLOTYRE'].pct_change()
fresh['BERGEPAINT'] = freshdf['BERGEPAINT'].pct_change()
fresh['CASTROLIND'] = freshdf['CASTROLIND'].pct_change()
fresh['CUMMINSIND'] = freshdf['CUMMINSIND'].pct_change()
fresh['DHFL'] = freshdf['DHFL'].pct_change()
fresh = fresh.drop([0], axis = 0)
sns.pairplot(fresh.dropna())

##2.7
#fresh['APOLLO roll avg'] = fresh['APOLLOTYRE'].rolling(7, win_type=None).mean()
fresh['APOLLO roll std'] = fresh['APOLLOTYRE'].rolling(7, win_type=None).std()
#print(fresh['APOLLO roll std'])
plt.plot(fresh['APOLLO roll std'])

##2.8
nifty = pd.read_csv('D:\\susmi\\stocks\\Nifty50.csv')
nifty['Volatility'] = nifty['Close'].pct_change()
print(nifty['Volatility'].head())
nifty['Volatility'] = nifty['Volatility'].rolling(7, win_type=None).std()
plt.plot(nifty['Volatility'])

##2.9
df_csv['SMA21'] = df_csv['Close Price'].rolling(21, win_type=None).mean()
df_csv['SMA34'] = df_csv['Close Price'].rolling(34, win_type=None).mean()
plt.plot(df_csv['SMA34'], label='34 day')
plt.plot(df_csv['SMA21'], label='21 day')
plt.plot(df_csv['Average Price'], label='Average Price')
plt.gca().legend()
#decide a call

##2.10 - bollinger bands
df_csv['average'] = df_csv['Close Price'].rolling(14, win_type=None).mean()
df_csv['std'] = df_csv['Close Price'].rolling(14, win_type=None).std()
df_csv['Upper band'] = df_csv['average'] + 2*df_csv['std']
df_csv['Lower band'] = df_csv['average'] - 2*df_csv['std']
plt.plot(df_csv['Date'], df_csv['Upper band'], df_csv['Date'], df_csv['average'], df_csv['Date'], df_csv['Lower band'], df_csv['Date'], df_csv['Average Price'])
plt.gca().legend(('Upper band', 'Average', 'Lower band', 'Daily Average'))

#df_csv.to_csv('week3.csv')