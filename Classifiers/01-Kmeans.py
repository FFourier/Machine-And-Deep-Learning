#Description
# The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the 
# inertia or within-cluster sum-of-squares (see below). This algorithm requires the number of clusters to be specified. 
# It scales well to large number of samples and has been used across a large range of application areas in many different fields.

#! Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

#! Preparing data

#* Loading the dataset
dataset=pd.read_csv('cars.csv',low_memory=False)
# https://www.kaggle.com/abineshkumark/carsdata

#* Selecting inputs columns
x=dataset.iloc[:,0].values ##cylinders
y=dataset.iloc[:,6].values #Year
x=x.reshape(len(x),1)

#* Mapping brand names into int values
y_real=pd.factorize(dataset.iloc[:,7].values)

#! Model 

#* Selection of cluster number based on the elbow curve
km=KMeans(n_clusters=3)

#* Training the model   
y_predicted = km.fit_predict(x,y)
   
#* Obtaining model accuracy, comparing the original values and predicted ones
accuracy=metrics.adjusted_rand_score(dataset.iloc[:,7].values,y_predicted)
print(f'\nAccuracy: {accuracy} \nIterations: {km.n_iter_}\n')

#! Visualization

fig, axs = plt.subplots(1,3)
fig.set_size_inches(18.5, 5)

axs[0].scatter(x,y,c=y_real[0],alpha=0.5)
axs[0].set_title('Dataset')


kmeans = [KMeans(n_clusters=i) for i in range(1, 10)]
score = [kmeans[i].fit(x).score(x) for i in range(len(kmeans))]
axs[1].plot(range(1, 10),score)
axs[1].set_title('Elbow Curve')

axs[2].scatter(x,y,c=y_predicted,s=30,alpha=0.6)
axs[2].set_title('K-Means')

plt.show()