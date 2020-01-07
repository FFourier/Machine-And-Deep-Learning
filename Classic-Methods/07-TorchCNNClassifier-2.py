#! Libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim

import matplotlib.pyplot as plt

from tqdm import tqdm, trange


#! Data

#* Bringing data from a preprocesed numpy object
full_dataset=np.load("training_data.npy",allow_pickle=True)
#? This part will be explained in a next python file.

#* Splitting the dataset in train and validation set
train_set, val_set = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.75)+1, int(len(full_dataset)*0.25)])

#* Taking training inputs and ouputs in different tensors
X = torch.Tensor([i[0] for i in train_set])
Y = torch.Tensor([i[1] for i in train_set])

#* Taking validation inputs and ouputs in different tensors
X_a = torch.Tensor([i[0] for i in val_set])
Y_a = torch.Tensor([i[1] for i in val_set])

#! Convolutional neural network architecture

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)
        
        self.fc1=nn.Linear(128*2*2, 170, bias=True)
        self.fc2=nn.Linear(170, 56, bias=True)
        self.fc3=nn.Linear(56, 2, bias=True)
    
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = torch.flatten(x, start_dim=1)
        x= F.relu(self.fc1(x)) 
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        return F.softmax(x, dim=1)

#* Instancing the network
net_class=Net()

#! Training loop

BATCH_SIZE = 100
EPOCHS = 3

optimizer = optim.Adam(net_class.parameters(),lr= 0.0001)
loss_function = nn.MSELoss()

loss_grap=[]

for epoch in range(EPOCHS):
    loss_mean=0
    for i in trange(0, len(train_set), BATCH_SIZE,  desc='Loss calculation'):
        #The iterating way is trying to emulate a data loader.
        
        batch_X=X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y=Y[i:i+BATCH_SIZE]
         #For this reason, each data batch is stored on those tensors

        net_class.zero_grad()
        outputs = net_class(batch_X)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()
        loss_grap.append(loss)
        loss_mean+=loss
    print(f'Epoch {epoch} error mean {loss_mean/len(train_set)}')


#! Accuracy

correct = 0
total = 0
with torch.no_grad():
    
    for i in trange(len(X_a), desc='Accuracy calculation'):
        
        real_class = torch.argmax(Y_a[i])
        
        net_out = net_class(X_a[i].view(-1,1,50,50))[0]
        
        predicted_class = torch.argmax(net_out)
        
        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy: ",round(correct/total,3))


#! Visualization


plt.plot(range(len(loss_grap)),loss_grap)
plt.title('Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()