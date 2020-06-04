# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:17:57 2019

@author: Susmita Mathew
"""

import pandas as pd

df_csv = pd.read_csv("RELINFRA.csv")

##1.1
for x in df_csv['Series']:
    if x != 'EQ':
        df_csv.drop([x], axis = 0)

##1.2
print(df_csv.shape)
print(df_csv.iloc[404:, 8].max())
print(df_csv.iloc[404:, 8].min())
print(df_csv.iloc[404:, 8].mean())

##1.3
print(df_csv['Date'].dtype)
df_csv['Date'] = df_csv['Date'].astype('datetime64[ns]')
print(df_csv['Date'].dtype)

print(df_csv['Date'].max())
print(df_csv['Date'].min())
print(df_csv['Date'].max() - df_csv['Date'].min())

##1.4
month = pd.DatetimeIndex(df_csv['Date']).month
df_csv['Month'] = month
year = pd.DatetimeIndex(df_csv['Date']).year
df_csv['Year'] = year
#To calculate vwap:
num = df_csv['Average Price']*df_csv['Total Traded Quantity']
df_csv['Numerator'] = num
vwap = df_csv.groupby(['Year','Month']).sum()
vwap['VWAP'] = vwap['Numerator']/vwap['Total Traded Quantity']
#vwap = df_csv.groupby(['Year','Month']).sum().groupby(level=[0]).cumsum()
print(vwap)

##1.5
def calcAvg(n):
    res = df_csv.iloc[(493-n):,8].mean()
    return res

def calcPerc(n):
    a = df_csv.iloc[493-n,8]
    b = df_csv.iloc[493,8]
    res = ((abs(a-b))/a)*100
    return res

print("Last 1 week - Average Price: ", calcAvg(7), "Profit/Loss Percentage:" ,calcPerc(7))
print("Last 2 weeks - Average Price: ", calcAvg(14), "Profit/Loss Percentage:" ,calcPerc(14))
print("Last 1 month(30 days) - Average Price: ", calcAvg(30), "Profit/Loss Percentage:" ,calcPerc(30))
print("Last 3 months(90 days) - Average Price: ", calcAvg(90), "Profit/Loss Percentage:" ,calcPerc(90))
print("Last 6 months(180 days) - Average Price: ", calcAvg(180), "Profit/Loss Percentage:" ,calcPerc(180))
print("Last 1 year(365 days) - Average Price: ", calcAvg(365), "Profit/Loss Percentage:" ,calcPerc(365))

##1.6
df_csv['Day_Perc_Change'] = df_csv['Close Price'].pct_change()
df_csv.loc[0,'Day_Perc_Change'] = 0
print(df_csv.head())

##1.7
df_csv['Trend'] = 0
for p in range(0,494):
    n = df_csv.at[p,'Day_Perc_Change']
    n *= 100
    if n>=-0.5 and n<=0.5:
        df_csv.loc[p,'Trend'] = "Slight or No change"
    elif n>0.5 and n<1:
        df_csv.loc[p,'Trend'] = "Slight positive"
    elif n>=-1 and n<=-0.5:
        df_csv.loc[p,'Trend'] = "Slight negative"
    elif n>=1 and n<=3:
        df_csv.loc[p,'Trend'] = "Positive"
    elif n>=-3 and n<=-1:
        df_csv.loc[p,'Trend'] = "Negative"
    elif n>=3 and n<=7:
        df_csv.loc[p,'Trend'] = "Among top gainers"
    elif n>=-7 and n<=-3:
        df_csv.loc[p,'Trend'] = "Among top losers"
    elif n>=7:
        df_csv.loc[p,'Trend'] = "Bull run"
    elif n<=-7:
        df_csv.loc[p,'Trend'] = "Bear drop"

##1.8
trendAvg = df_csv['Total Traded Quantity'].groupby(df_csv['Trend']).mean()
trendMed = df_csv['Total Traded Quantity'].groupby(df_csv['Trend']).median()
print("Averages of Total Traded Quantity")
print(trendAvg)
print("Medians of Total Traded Quantity")
print(trendMed)

df_csv.to_csv('week2.csv')
