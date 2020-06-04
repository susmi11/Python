# -*- coding: utf-8 -*-
"""
Created on Sat Aug 31 07:54:41 2019

@author: Susmita Mathew

Problem Statements
6.1 Create a table/data frame with the closing prices of 30 different stocks, with 10 from each of the caps
6.2 Calculate average annual percentage return and volatility of all 30 stocks over a theoretical one year period
6.3 Cluster the 30 stocks according to their mean annual Volatilities and Returns using K-means clustering. Identify the optimum number of clusters using the Elbow curve method
6.4 Prepare a separate Data frame to show which stocks belong to the same cluster 
    
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import stdev
sns.set_style('whitegrid')
from os import listdir
from os.path import isfile, join, isdir
from sklearn.cluster import KMeans 
from sklearn import metrics 
from scipy.spatial.distance import cdist 

##6.1
main_df = pd.DataFrame()

def getAllFilesRecursive(root):
    files = [ join(root,f) for f in listdir(root) if isfile(join(root,f))]
    dirs = [ d for d in listdir(root) if isdir(join(root,d))]
    for d in dirs:
        files_in_d = getAllFilesRecursive(join(root,d))
        if files_in_d:
            for f in files_in_d:
                files.append(join(root,f))
    return files

f = getAllFilesRecursive('D:\susmi\stocks\sixth_mod')    

for file in f:
    df1 = pd.read_csv(file)
    x = df1['Symbol'][1]
    main_df[x] = df1['Close Price']
    
##6.2
#no. of working days in a year = 253
#volatility as std deviation
vol = []
avgAnn = []
for company in main_df:
    sample = []
    for i in range(0,254):
        sample.append(main_df[company][i])
    v = stdev(sample)
    vol.append(v)
    a = ((sample[0]-sample[253])/sample[0])*100
    avgAnn.append(a)        

#visualizing given data    
plt.scatter(vol, avgAnn)
plt.show()

##6.3
X = np.array(list(zip(vol, avgAnn)))

distortions = [] 
inertias = [] 
mapping1 = {} 
mapping2 = {} 
K = range(1,10)
for k in K: 
	#Building and fitting the model 
	kmeanModel = KMeans(n_clusters=k).fit(X) 
	kmeanModel.fit(X)	 
	
	distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
					'euclidean'),axis=1)) / X.shape[0]) 
	inertias.append(kmeanModel.inertia_) 

	mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 
				'euclidean'),axis=1)) / X.shape[0] 
	mapping2[k] = kmeanModel.inertia_ 

#using distortion
for key,val in mapping1.items(): 
    print(str(key)+' : '+str(val))
plt.plot(K, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 

#using inertia
for key,val in mapping2.items(): 
	print(str(key)+' : '+str(val)) 
plt.plot(K, inertias, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show() 

#considering both inertia and distortion, choosing k as 4
kmeans = KMeans(n_clusters=4).fit(X)
l =list(kmeans.labels_)
#comment below line for clearer soln of 6.4
plt.scatter(vol, avgAnn, c = l)

##6.4
companies = []
for i in main_df:
    companies.append(i)

c1 = [None]*30
c2 = [None]*30
c3 = [None]*30
c4 = [None]*30
c=0
for i in range(0,30):
    if(l[i]==0):
        c1[i] = companies[i]
        c += 1
    elif(l[i]==1):
        c2[i] = companies[i]
        c += 1
    elif(l[i]==2):
        c3[i] = companies[i]
        c += 1
    else:
        c4[i] = companies[i]
        c += 1

print(c1)
d = dict( Cluster_1 = np.array(c1), Cluster_2 = np.array(c2), Cluster_3 = np.array(c3), Cluster_4 = np.array(c4) )
new_df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in d.items() ]))
print(new_df.head())
new_df['Volatility'] = vol
new_df['Average annual ret'] = avgAnn

for i in range(0,30):
    a = new_df.Cluster_1[i]
    if(a != None):
        p = plt.scatter(new_df.Volatility[i], new_df['Average annual ret'][i], s=100, alpha = 0.3, c='red')
for i in range(0,30):
    a = new_df.Cluster_2[i]
    if(a != None):
        q = plt.scatter(new_df.Volatility[i], new_df['Average annual ret'][i], s=100, alpha = 0.3, c='green')
for i in range(0,30):
    a = new_df.Cluster_3[i]
    if(a != None):
        r = plt.scatter(new_df.Volatility[i], new_df['Average annual ret'][i], s=100, alpha = 0.3, c='blue')
for i in range(0,30):
    a = new_df.Cluster_4[i]
    if(a != None):
        s = plt.scatter(new_df.Volatility[i], new_df['Average annual ret'][i], s=100, alpha = 0.3, c='yellow')
plt.legend(handles = [p,q,r,s], labels = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])

for i in range(0,30):
    plt.annotate(companies[i], (vol[i], avgAnn[i]))