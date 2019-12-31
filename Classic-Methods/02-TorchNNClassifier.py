#Libraries

import torch
import torch.nn as nn
import torch.optim  as optim
import torch.nn.functional as F

import numpy as np

import torchvision
from torchvision import transforms, datasets

from tqdm import tqdm

import matplotlib.pyplot as plt

from skimage import transform,io


#MNIST dataset splited and shuffled
train=datasets.MNIST("",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

test=datasets.MNIST("",train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))

trainset= torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset= torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)


#Neural network architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        #784 es the flatten vector dimension of the 28x28 image
        self.fc1=nn.Linear(784, 64, bias=True)
        self.fc2=nn.Linear(64, 64, bias=True)
        self.fc3=nn.Linear(64, 64, bias=True)
        self.fc4=nn.Linear(64, 10, bias=True)
        #10 outputs its due to ten digits.
    
    def forward(self,x):

        #Each layer has a relu as activation function
        x= F.relu(self.fc1(x)) 
        x= F.relu(self.fc2(x))
        x= F.relu(self.fc3(x)) 
        x= self.fc4(x)

        #The output is a softmax function
        return F.softmax(x, dim=1)

#Instance of the NN architecture
net=Net()

#Loading the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS=3

for epoch in range(EPOCHS):
    for data in tqdm(trainset):
        X,y=data
         #Spliting the trainset into inputs(X) and outputs(y)
        net.zero_grad()
        output=net(X.view(-1,28*28))
            #Pass the data throught the network
        
        loss = F.nll_loss(output, y)
        
        loss.backward()
        
        optimizer.step()
            #This is what adjust the weights for us

    print(loss)
correct=0
total=0


with torch.no_grad():
    # We don't want to calculate the gradients, just test the network
    for data in tqdm(trainset):
        X,y=data
        output=net(X.view(-1,28*28))
        for idx,i in enumerate(output):
            if torch.argmax(i)==y[idx]:
                # argmax Returns the indices of the maximum values along an axis.
                correct +=1
            total+=1          
print('Accuracy: ',round(correct/total,3))

#Splitting the testset
for data in testset:
        x_test,y_test=data
        
#Plotting the result of the network with the trainset        
nx,ny,c=5,2,0
fig, axs = plt.subplots(ny,nx)
fig.set_size_inches(nx*2, 4)
for i in range(nx):
    for j in range(ny):
        prueba=x_test[c][0]
        output_prueba=net(prueba.view(1,28*28))
        _,salida=torch.max(output_prueba,1)
        axs[j,i].imshow(prueba, cmap="gray")
        axs[j,i].axis('off')
        axs[j,i].set_title(str(salida))
        c+=1
plt.show()


#Image preparation


# read in grey-scale
grey = io.imread('try.png', as_gray=True)
# resize to 28x28
small_grey = transform.resize(grey, (28,28), mode='symmetric', preserve_range=True)

input_img = torch.Tensor([i for i in small_grey])

output_try=net(input_img.view(1,28*28))

idx,salida=torch.max(output_try,1)

#Font 
font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

#Plotting the network result with an external image   
plt.figure(figsize = (3,3))
plt.imshow(input_img, cmap='gray')
plt.title(str(salida),  fontdict=font)
plt.axis('off')
plt.show()