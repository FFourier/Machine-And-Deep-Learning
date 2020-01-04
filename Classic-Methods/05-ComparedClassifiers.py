#! Libraries

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.tree import export_graphviz

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from subprocess import call

from IPython.display import Image

import pydot

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

import statistics as st

#! Data

#* Loading dataset
iris =load_iris()

#* Converting dataset to pandas dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)

#* Adding species column to dataframe
df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)

#* Removing rows with virginica value
df=df[df.species != 'virginica']

#* Removing width columns from dataframe
df=df.drop('sepal width (cm)', axis=1)
df=df.drop('petal width (cm)', axis=1)

#* Reseting index and mixing dataframe
df = df.sample(frac=1).reset_index(drop=True)

#* Extracting data from the dataframe
x, y=df.iloc[:,0].values, df.iloc[:,1].values
x,y=x.reshape(len(x),1), y.reshape(len(y),1)

#* Organizing data in an array
data=np.zeros([len(x),2])
data[:len(x),0]=x[:,0]
data[:len(x),1]=y[:,0]

#! K - Means

#! Elbow curve

#* Evualating Kmeans with different number of clusters
kmeans = [KMeans(n_clusters=i) for i in range(1, 5)]

#* Obtaining its value and stacking in a list
score = [kmeans[i].fit(data).score(data) for i in range(len(kmeans))]

#* Model implementation with the number of cluster determinated with the curve
km=KMeans(n_clusters=2)
output_km = km.fit_predict(data)

#* Copying data frame  for maintain primary data saved
compare=df.copy()

#* Assigning labels to results 
output_km_label=['setosa' if output_km[i] == 0 else 'versicolor' for i in range(len(output_km))]

if not df.species[0] == output_km_label[0]:
    output_km_label=['versicolor' if output_km[i] == 0 else 'setosa' for i in range(len(output_km))]
    # This IF sentence is important because k-means is an unsupervised learning algorithm, so it didn't really knows if the label is right

#* Stacking a column with K-Means result
compare['Cluster']=output_km_label

accuracy=metrics.adjusted_rand_score(compare.iloc[:,2].values,output_km)


#! Random Forest

#* Creating Test and Train data
compare['is_train']=np.random.uniform(0,1,len(compare))<=.75
    # We are creating a random number between 0 and 1 for each row, if that number is less tan 0.75 will be true, then the 75% of data will be selected for train.

#* Creating dataframes with test rows and training rows
train,test = compare[compare['is_train']==True],compare[compare['is_train']==False]

#* Deleting the column to keep the dataframe unchanged
compare = compare.drop('is_train', 1)

#! Model 

#* Creating a random forest classifier
clf=RandomForestClassifier(n_estimators=100,n_jobs=2,random_state=0)

#* Selecting features
features = compare.columns[:2]

#* Training the classifier
clf.fit(train[features],pd.factorize(train['species'])[0])

#* Applying the trained Classifier to the test
output_RF_t=iris.target_names[clf.predict(test[features])]

#* Confussion matrix
pd.crosstab(test['species'],output_RF_t,rownames=['Actual Species'],colnames=['Predicted Species'])

#* Adding column with random forest results
compare['RandForest']=iris.target_names[clf.predict(compare[features])]

#* Getting estimator for the graph 
estimator = clf.estimators_[5]


#* Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = [iris.feature_names[0],iris.feature_names[2]],
                class_names = iris.target_names[:2],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

#* Display in jupyter notebook
img = mpimg.imread('tree.png')

output_RF=(pd.factorize(compare['RandForest']))[0]



#! Logistic Regression

#! Data

#* Samples number
n=len(data)
#* Labels
h=n//2
dimen=2

#* Splitting the data in inputs (x) and outputs (y)
x=torch.from_numpy(data).float().requires_grad_(True)
y=(torch.from_numpy((pd.factorize(df['species']))[0]).view(len(data),1)).float()

#! Model

#* Building the model
model= nn.Sequential(nn.Linear(2,1), nn.Sigmoid())

#* Loss function and optimizer method
loss_function= nn.BCELoss()
optimizer=optim.Adam(model.parameters(),lr=0.025)

#! Training loop

losses=[]
iterations=400

for i in tqdm(range(iterations)):
    
    result=model(x)
    loss=loss_function(result,y)
 
    losses.append(loss.data)

    optimizer.zero_grad()
        
    loss.backward()
        
    optimizer.step()

#* Passing data through the model
prediction=model(x)

#* List with the corresponding labels
prediction=['purple' if prediction[i] < 0.5 else 'yellow' for i in range(len(prediction))]

#* weights
w = list(model.parameters())

w0 = w[0].data.numpy()

#! Visualization

#* Parameters to plot the line
x_axis = np.linspace(np.min(data[:,0]), np.max(data[:,0]), len(x))
y_axis = -(w[1].data.numpy() + x_axis*w0[0][0]) / w0[0][1]
    
#* Font format   
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 8,
        }

#! FINAL PLOTTING

#* Plotting
fig, axs = plt.subplots(2,4, constrained_layout=True)
fig.suptitle('Comparing classification methods', fontsize=30)
fig.set_size_inches(16, 8)

axs[0][0].scatter(data[:,0],data[:,1],c='green',alpha=0.6,s=65)
axs[0][0].set_title('Dataset')

axs[0][1].scatter(data[:,0],data[:,1],alpha=0.6,s=65,c=pd.factorize(df['species'])[0])
axs[0][1].set_title('Dataset wiht outputs')


axs[0][2].plot(range(1, 5),score)
axs[0][2].set_xlabel('Number of Clusters')
axs[0][2].set_ylabel('Score')
axs[0][2].set_title('Elbow Curve')

axs[0][3].scatter(data[:,0],data[:,1],c=output_km,alpha=0.6,s=65)
axs[0][3].scatter([km.cluster_centers_[0][0],km.cluster_centers_[1][0]],[km.cluster_centers_[0][1],km.cluster_centers_[1][1]],marker="X",s=150,c='red')
axs[0][3].set_title('K means')

axs[1][0].imshow(img)
axs[1][0].set_title('Random Forest')
axs[1][0].axis('off')

axs[1][1].scatter(data[:,0],data[:,1],c=output_RF,alpha=0.6,s=65)
axs[1][1].set_title('Random Forest Classifier')

axs[1][2].plot(range(iterations),losses)
axs[1][2].set_title('Loss')

axs[1][3].plot(x_axis, y_axis,'g--')    
for i in range(len(x)):
    axs[1][3].scatter(x[i,0].data, x[i,1].data,s=55,alpha=0.7,c=prediction[i])
axs[1][3].set_title('Logistic Regression')

plt.show()