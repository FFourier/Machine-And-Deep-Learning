import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston


boston=load_boston()
    #Loading the dataset

x=np.array(boston.data[:,5])
    #Input matrix
        #We take the 5th row, that represents the average number of rooms

Y=np.array(boston.target)
    #Median value of owner-occupied homes in $1000's

print(f'\nx:         {np.size(x)}\nones(506): {np.size(np.ones(506))}')
    #Validation of sizes
X=np.array([np.ones(506),x]).T
    #Addition of a row full of ones for the independents factors 
print(f'X:         {X.shape}\n\n')

B=np.linalg.inv(X.T @ X) @ X.T @ Y
    #Implementation of the derivative concerned to the mean square error

print(f'X:   {X.shape}\nX.T: {X.T.shape}\nY:   {np.size(Y)}\nB:   {np.size(B)}\n')
print(B)
    #Dimensions summary and final result

plt.scatter(x,Y,alpha=0.5)
plt.plot([4,9],[B[0]+B[1]*4,B[0]+B[1]*9],c='red')
plt.title('Linear Regression')
plt.show()


###

# Implimentation with scklearn

###

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,Y)
b=regressor.intercept_
m=regressor.coef_[1] 
plt.scatter(x,Y,alpha=0.5)
plt.plot([4,9],[b+m*4,b+m*9],c='black')
plt.title('Regression with sklearn')
plt.show()

