#! Libraries

import numpy as np
from tqdm import tqdm
from skimage import transform,io

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim

from torchvision import utils
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt

#! Preparing data

#* MNIST dataset splited and shuffled
train=datasets.MNIST("",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))

test=datasets.MNIST("",train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))

trainset= torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
testset= torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)


#! Convolutional Neural network architecture

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #* Two convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        #* Three fully conected layers
        self.fc1=nn.Linear(64*4*4, 341, bias=True)
        self.fc2=nn.Linear(341, 113, bias=True)
        self.fc3=nn.Linear(113, 10, bias=True)
        #* 10 outputs due to ten digits.
    
    def forward(self,x):
        #* Each convolutional layer needs a relu function as activation function and a pooling to decreese data
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))

        #* It's necesary to flat the tensor to pass it through the linear layers
        x = torch.flatten(x, start_dim=1)

        #* Fully conected layers with relu as activation function
        x= F.relu(self.fc1(x)) 
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        #* Introducing softmax function
        return F.softmax(x, dim=1)

#* Instancing the network
net_class=Net()

#* Loading the optimizer
optimizer = optim.Adam(net_class.parameters(), lr=0.001)


#! Training loop

EPOCHS=3

for epoch in range(EPOCHS):
    for data in tqdm(trainset, desc='Loss calculation'):
        #* Spliting trainset into inputs(X) and outputs(y)
        X,y=data
        net_class.zero_grad()
        output=net_class(X.view(-1,1,28,28))
        # The main difference between the training loop of a convolutional NN and a regular NN, 
        # is that in this you pass the tensor with its dimensions, cuz the first layers are convolutional
        
        loss = F.nll_loss(output, y)
        
        loss.backward()
        
        optimizer.step()

    print(loss)


#! Forward pass

correct=0
total=0

with torch.no_grad():
    for data in tqdm(trainset, desc='Accuracy calculation'):
        X,y=data
        output=net_class(X.view(-1,1,28,28))
        for idx,i in enumerate(output):
            if torch.argmax(i)==y[idx]:
                correct +=1
            total+=1
            
print('Accuracy: ',round(correct/total,3))


#! Visualization

grey = io.imread('try.png', as_gray=True)

small_grey = transform.resize(grey, (28,28), mode='symmetric', preserve_range=True)

input_img = torch.Tensor([i for i in small_grey])

out=net_class(input_img.view(1,1,28,28))
idx, salida=torch.max(out,1)

plt.imshow(input_img, cmap='gray')
plt.title(str(salida))
plt.axis('off')
plt.show()