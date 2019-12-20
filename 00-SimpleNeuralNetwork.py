import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return (np.exp(-x))/(1+np.exp(-x))**2

training_inputs=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
training_outputs=np.array([[0],[1],[1],[0]])

np.random.seed(1)

synaptic_weights=-(-1-1)*np.random.random((3,1))-1
    #-(b - a) * random_sample() + a   [a:b]
    
print('Random starting synaptic weights: \n',synaptic_weights)

for iteration in range(1000):
    
    input_layer=training_inputs
    
    rrs=np.dot(input_layer,synaptic_weights) #Multiplication and addition
    
    outputs=sigmoid(rrs) #Calculation of the sigmoid function
        
    error=training_outputs - outputs #Calculation of the error              
    
    adjustments=error*sigmoid_derivative(outputs)
    
    synaptic_weights+=np.dot(input_layer.T,adjustments)

print('\nSynaptic weights after training:\n ',synaptic_weights)
print('\nOutputs after {} training is: \n{}'.format(iteration,outputs))