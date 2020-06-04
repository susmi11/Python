# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 00:50:03 2019

@author: Susmita Mathew
"""
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

##5.1
df = pd.read_csv('D:\susmi\stocks\CUMMINSIND.csv')
print(df.head())
df['daily return'] = (df['Close Price']-df['Open Price'])/df['Open Price']
dailyret_mean = df['daily return'].mean()
std_dev = df['daily return'].std()
annual_mean = dailyret_mean*252
annual_stdev = std_dev*math.sqrt(252)
print("Annual mean", annual_mean)
print("Annual standard deviation", annual_stdev)

##5.2
df2 = pd.read_csv('D:\susmi\stocks\DHFL.csv')
df2['daily return'] = (df2['Close Price']-df2['Open Price'])/df2['Open Price']
dailyret_mean2 = df2['daily return'].mean()
std_dev2 = df2['daily return'].std()
annual_mean2 = dailyret_mean2*252
annual_stdev2 = std_dev2*math.sqrt(252)
df3 = pd.read_csv('D:\susmi\stocks\APOLLOTYRE.csv')
df3['daily return'] = (df3['Close Price']-df3['Open Price'])/df3['Open Price']
dailyret_mean3 = df3['daily return'].mean()
std_dev3 = df3['daily return'].std()
annual_mean3 = dailyret_mean3*252
annual_stdev3 = std_dev3*math.sqrt(252)
df4 = pd.read_csv('D:\susmi\stocks\CASTROLIND.csv')
df4['daily return'] = (df4['Close Price']-df4['Open Price'])/df4['Open Price']
dailyret_mean4 = df4['daily return'].mean()
std_dev4 = df4['daily return'].std()
annual_mean4 = dailyret_mean4*252
annual_stdev4 = std_dev4*math.sqrt(252)
df5 = pd.read_csv('D:\susmi\stocks\BERGEPAINT.csv')
df5['daily return'] = (df5['Close Price']-df5['Open Price'])/df5['Open Price']
dailyret_mean5 = df5['daily return'].mean()
std_dev5 = df5['daily return'].std()
annual_mean5 = dailyret_mean5*252
annual_stdev5 = std_dev5*math.sqrt(252)
#annual return and volatility of portfolio
annual_ret = 0.2*annual_mean + 0.2*annual_mean2 + 0.2*annual_mean3 + 0.2*annual_mean4 + 0.2*annual_mean5 
volatility = 0.2*annual_stdev + 0.2*annual_stdev2 + 0.2*annual_stdev3 + 0.2*annual_stdev4 + 0.2*annual_stdev5

##5.3
returns = []
volatility = []
scolor = []
low_vol = 100
hi_sharpe = -100
for a in np.arange(0, 1.1, 0.1):
    for b in np.arange(0,1.1-a, 0.1):
        for c in np.arange(0,1.1-(a+b), 0.1):
            for d in np.arange(0,1.1-(a+b+c), 0.1):
                e = 1 - (a+b+c+d)
                ret = a*annual_mean + b*annual_mean2 + c*annual_mean3 + d*annual_mean4 + e*annual_mean5
                vol = a*annual_stdev + b*annual_stdev2 + c*annual_stdev3 + d*annual_stdev4 + e*annual_stdev5
                if vol<low_vol:
                    low_vol = vol
                    low_y = ret
                sharpe = ret/vol
                if sharpe>hi_sharpe:
                    hi_y = ret
                    hi_x = vol
                returns.append(ret)
                volatility.append(vol)
                scolor.append(sharpe)
y = scolor
plt.scatter(volatility,returns, c=y)
plt.xlabel('Volatility')
plt.ylabel('Returns')
##5.4
plt.plot(low_vol,low_y,color = 'red', marker = '*')
plt.plot(hi_x, hi_y,color = 'violet', marker = '*')
