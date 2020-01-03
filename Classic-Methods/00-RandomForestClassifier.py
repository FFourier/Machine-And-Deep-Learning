# Description
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset 
# and uses averaging to improve the predictive accuracy and control over-fitting. 

#! Libraries

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from subprocess import call
from sklearn.tree import export_graphviz
from IPython.display import Image

#! Preparing data

#* Specifies seed for random number
np.random.seed(0)
    
#* Creating an object called iris with the iris data
iris =load_iris()
    
#* Creating a data frame with the four feature variables
df = pd.DataFrame(iris.data, columns=iris.feature_names)
    
#* Adding a new column for the species name   
df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)
    

#* Creating Test and Train data
df['is_train']=np.random.uniform(0,1,len(df))<=.75
    #? We are creating a random number between 0 and 1 for each row, if that number is less tan 0.75 will be true, then the 75% of data will be selected for train.

#* Creating dataframes with test rows and training rows
train,test = df[df['is_train']==True], df[df['is_train']==False]
    #? If is_train is true will be part of the training data

#* Create a list of the feature column's names
features = df.columns[:4]
    #? Storing the first four columns

#* Converting each species name into digits
y=pd.factorize(train['species'])[0]

#! Model

#* Creating a random forest classifier
clf=RandomForestClassifier(n_estimators=100,n_jobs=2,random_state=0)
    #? n_estimators=The number of trees in the forest.
    #? n_jobs means the nombre of jobs to run in parallel

#* Training the classifier
clf.fit(train[features],y)
    #? Recibes as parameter the two column data and the related labels

#* Applying the trained Classifier to the test
clf.predict(test[features])

#* Distribution of prabilities
clf.predict_proba(test[features])[10:20]
    #? Predict probability,those trhee columns mean three leaf notes at the end
    # for example the firt value means 1 for setosa, 0 for versicolor, 0 for virginica
        # of course, the sum of three values is equal to 1
        # in case two features have equal probability, it will choose the first

#* mapping names for the plants for each predicted plant class
preds=iris.target_names[clf.predict(test[features])]

#* Creating confusion matrix
print(pd.crosstab(test['species'],preds,rownames=['Actual Species'],colnames=['Predicted Species']))


#! Visualization

#* Extract single tree
estimator = clf.estimators_[5]

#* Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = features,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

#* Convert to png using system command (requires Graphviz)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])