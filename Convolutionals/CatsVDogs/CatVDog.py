
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

from tqdm import tqdm, trange

import cv2 

#! --------------
#! CUDA
#! --------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\nGPU Available\n")
else:
    device = torch.device("cpu")
    print("CPU Available\n")
    

#! --------------
#! External data
#! --------------


path1 = r'imgs\gat.jpg'
path2 = r'imgs\salchi.jpg'
path3 = r'imgs\malvada.jpeg'
path4 = r'imgs\charli.jpeg'
path5 = r'imgs\bartolo.jpeg'
path6 = r'imgs\boz.jpg'

img = cv2.imread(path6, cv2.IMREAD_GRAYSCALE)

plt.imshow(img,cmap='gray')
plt.title('Test Image')
plt.axis('off')
plt.show()
print('')

img = cv2.resize(img, (50, 50))


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
#! Load or Create
#! --------------

net_class = Net()

load = False

if load:

    net_class.load_state_dict(torch.load(r'NNCVD.pt'))
    net_class = net_class.to(device)
    print('Model loaded\n')
    create = False
    

else:
    
    print('Creating a model\n')
    net_class=net_class.to(device)
    create = True

    
if create:
    
#! Data
    full_dataset=np.load(r'training_data.npy',allow_pickle=True)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [int(len(full_dataset)*0.75)+1, int(len(full_dataset)*0.25)])

    X = torch.Tensor([i[0] for i in train_set])
    Y = torch.Tensor([i[1] for i in train_set])

    X_a = torch.Tensor([i[0] for i in val_set])
    Y_a = torch.Tensor([i[1] for i in val_set])

#! Training loop
    
    print('\n\nTraining Loop\n')
    BATCH_SIZE = 100
    EPOCHS = 30

    optimizer = optim.Adam(net_class.parameters(),lr= 0.0001)
    loss_function = nn.MSELoss()

    loss_grap=[]
    mean_grap=[]

    for epoch in range(EPOCHS):
        loss_mean=0
        for i in trange(0, len(train_set), BATCH_SIZE,  desc='Loss calculation'):

            batch_X=X[i:i+BATCH_SIZE].view(-1,1,50,50)
            batch_y=Y[i:i+BATCH_SIZE]

            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            net_class.zero_grad()
            outputs = net_class(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            loss_grap.append(loss)
            loss_mean+=loss
        print(f'Epoch {epoch+1} error mean {(((loss_mean/i).item())*100):.5f}')
        mean_grap.append(((loss_mean/i).item())*100)

#! Plotting
    plt.plot(range(len(loss_grap)),loss_grap,c='black')

    xpt=np.linspace(0,len(loss_grap),len(mean_grap))
    xpt=[int(i) for i in xpt]

    plt.plot(xpt,mean_grap,c='red',lw=3)

    plt.title('Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()

#! Accuracy
    print('\nAccuracy Loop\n')
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
#! Testing
#! --------------

innn = torch.Tensor([i for i in img])
innn = innn.to(device)
net_res = net_class(innn.view(-1,1,50,50))
print(f'\nHay una certeza del {((net_res[0][0])*100):.2f} % que sea gato y un {((net_res[0][1])*100):.2f} % que sea perro')