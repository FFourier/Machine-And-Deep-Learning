
#TODO Python file version of the notebook Cat vs Dog

#! --------------
#! Libraries
#! --------------

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim

import matplotlib.pyplot as plt

from tqdm import trange

import time

full_dataset=np.load(r'training_data.npy',allow_pickle=True)
train_set, val_set = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.75)+1, int(len(full_dataset)*0.25)])

X = torch.Tensor([i[0] for i in train_set])
Y = torch.Tensor([i[1] for i in train_set])

X_a = torch.Tensor([i[0] for i in val_set])
Y_a = torch.Tensor([i[1] for i in val_set])


#! --------------
#! Model
#! --------------


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
   

#! --------------
#! CUDA
#! --------------

    
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    net_class = Net().to(device)
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    net_class = Net()
    print("Running on the CPU")
    
print(f'\nClass architecture\n\n{net_class}')


#! --------------
#! Training Loop
#! --------------


BATCH_SIZE = 100
EPOCHS = 40

optimizer = optim.Adam(net_class.parameters(),lr= 0.0002)
loss_function = nn.MSELoss()

loss_grap=[]
mean_grap=[]
prin = True

print(f'\nTraining Loop\n')

start_time = time.time()

for epoch in range(EPOCHS):
    loss_mean=0
    for i in trange(0, len(train_set), BATCH_SIZE,  desc='Loss calculation'):
        
        batch_X=X[i:i+BATCH_SIZE].view(-1,1,50,50)
        batch_y=Y[i:i+BATCH_SIZE]
        
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        if prin:
            print(f'Input shape: {list(batch_X.shape)}')
        
        net_class.zero_grad()
        outputs = net_class(batch_X)
        
        if prin:
            print(f'\nOutput shape: {list(outputs.shape)}\t Y shape: {list(batch_y.shape)}\n')
            prin = False
                
        loss = loss_function(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        loss_grap.append(loss)
        loss_mean+=loss
    print(f'Epoch {epoch+1} error mean {(((loss_mean/i).item())*100):.5f}')
    mean_grap.append(((loss_mean/i).item())*100)

elapsed_time = time.time() - start_time

print(f'\nTraining time:\t[ {int(elapsed_time // 60)} : {int(elapsed_time % 60)} : {int(((elapsed_time % 60) % 1)*100)} ]')


#! --------------
#! Accuracy
#! --------------


correct = 0
total = 0
with torch.no_grad():
    
    for i in trange(len(X_a), desc='Accuracy calculation'):
        
        X_a,Y_a = X_a.to(device), Y_a.to(device)
        
        real_class = torch.argmax(Y_a[i])
        
        net_out = net_class(X_a[i].view(-1,1,50,50))[0]
        
        predicted_class = torch.argmax(net_out)
        
        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy: ",round(correct/total,3))


#! --------------
#! Save
#! --------------


sv = input(f'Do you want to save the model? [y|n]')

if sv == 'y':
    name = str(input(f'File name: '))
    name = name + '.pt'
    torch.save(net_class.state_dict(), name)
    print(f'Model [{name}] saved')
else:
    print('Model not saved')