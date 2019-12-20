import numpy as np
import matplotlib.pyplot as plt

func = lambda th: np.sin((1/2)*th[0]**2 - (1/4)*th[1]**2+3) * np.cos(2*th[0]+1-np.e**th[1])
    #It's a random function


res=100
    #Resolution: Means how much values _X, _Y will take

gp=2.5
    #Range of values

_X=np.linspace(-gp,gp,res)
_Y=np.linspace(-gp,gp,res)   
_Z = func(np.meshgrid(_X, _Y))
    #Function results

Theta=np.random.rand(2)*(gp-0.5)*2-(gp-0.5)
    #Random point in the surface

_T=np.copy(Theta)
start_point=np.copy(Theta)

#We are going to implement forward finite difference method wich is an aproximation of partial derivative
#where h is set as a fix value

h=0.001

lr=0.001
    #Learning rate

iterations=100000
    #Number of iterations

grad = np.zeros(2)
    #Gradiant vector initialization

for _ in range(iterations):

    for it,th in enumerate (Theta):
        
        _T = np.copy(Theta)

        _T[it] = _T[it] + h
        deriv = (func(_T) - func(Theta))/h
        grad[it] = deriv
            #f'(x) = ( f[x+h] - f[x]) / h ) 


    Theta -= lr*grad
        #The negative value of gradiant allows to find the minimum   
            
    
    if(_ %25 == 0):
        plt.plot(Theta[0],Theta[1],"o",c="red")
            #To see how its change, each 25 iterations will plot

plt.contourf(_X,_Y,_Z,25) #Surface
plt.colorbar() 
plt.plot(start_point[0],start_point[1],"o",c="white")   #Start point
plt.plot(Theta[0],Theta[1],"o",c="green") #Finish point
plt.title('Gradient Descent')
plt.show()