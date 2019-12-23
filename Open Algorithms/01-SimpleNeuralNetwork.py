import numpy as np

class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights=-(-1-1)*np.random.random((3,1))-1
            #Initialization of the synaptics weights

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_derivative(self, x):
        return (np.exp(-x))/(1+np.exp(-x))**2
            # It can be simplified as  x*(1-x)
    
    def train(self,training_inputs, training_outputs, training_iterations):
        
        for iteration in range(training_iterations):
            
            output=self.think(training_inputs)# Its a function that we're going to define later

            error=training_outputs-output #This error is needed for back propagation

            adjustments=np.dot(training_inputs.T, error*self.sigmoid_derivative(output))

            self.synaptic_weights+= adjustments

                #At this point, it's the same code as the 00
    
    def think(self,inputs):
        #With the network trained, you can call this method to implement it.

        inputs = inputs.astype(float)
        output=self.sigmoid(np.dot(inputs,self.synaptic_weights))

        return output
    
if __name__ == '__main__':
    neural_Network=NeuralNetwork()
    print(f'\nRandom synaptic weights: {neural_Network.synaptic_weights}')

    training_inputs=np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
    training_outputs=np.array([[0],[1],[1],[0]])
        #We´re using the same dataset of the 00

    n_iter=1000

    neural_Network.train(training_inputs,training_outputs,n_iter) 
    print(f'\nSynaptic weights after training: {neural_Network.synaptic_weights}')

    #Ask to user to provide custom inputs to test the neural network

    A=str(input('Input 1: '))
    B=str(input('Input 2: '))
    C=str(input('Input 3: '))

    print(f'\nNew situation: input data = [{A},{B},{C}]')
    print(f'\nOutput data: {neural_Network.think(np.array([A,B,C]))} with {n_iter} iterations')




