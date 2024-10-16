# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 : Choose the number of clusters (K): Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

STEP 2 : Initialize cluster centroids: Randomly select K data points from your dataset as the initial centroids of the clusters.

STEP 3 : Assign data points to clusters: Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

STEP 4 : Update cluster centroids: Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

STEP 5 : Repeat steps 3 and 4: Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Rohith T 
RegisterNumber: 212223040173  
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("Mall_Customers (1).csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.cluster import KMeans
wcss=[]
for i in range (1,11):
kmeans=KMeans(n_clusters = i,init="k-means++")
kmeans.fit(data.iloc[:,3:])
wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.xlabel("No. of clusters")
plt.ylabel("wcss")
plt.title("Elbow matter")
km=KMeans(n_clusters=5)
km.fit(data.iloc[:,3:])
y_pred=km.predict(data.iloc[:,3:])
y_pred
data["cluster"]=y_pred
df0=data[data["cluster"]==0]
df1=data[data["cluster"]==1]
df2=data[data["cluster"]==2]
df3=data[data["cluster"]==3]
df4=data[data["cluster"]==4]
plt.scatter(df0["Annual Income (k$)"],df0["Spending Scor
plt.scatter(df1["Annual Income (k$)"],df1["Spending Scor
plt.scatter(df2["Annual Income (k$)"],df2["Spending Scor
plt.scatter(df3["Annual Income (k$)"],df3["Spending Scor
plt.scatter(df4["Annual Income (k$)"],df4["Spending Scor
plt.legend()
plt.title("Customer Segmets")
```

## Output:
ELBOW GRAPH:
![Screenshot 2024-10-16 224006](https://github.com/user-attachments/assets/38864fb7-149c-4d4c-b104-f3d7fc4e0350)

PREDICTED VALUES:
![Screenshot 2024-10-16 224053](https://github.com/user-attachments/assets/ec9e7671-af50-45ed-ac9a-192842f0fec5)

FINAL GRAPH:
![Screenshot 2024-10-16 224107](https://github.com/user-attachments/assets/45196de1-6715-4326-862b-411f59b6244a)

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
