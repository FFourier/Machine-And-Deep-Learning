#TODO Python file version of the notebook Handwriten numbers 

#! --------------
#! Libraries
#! --------------

import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim  as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from skimage import transform,io

import cv2 

#! --------------
#! CUDA
#! --------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")


#! --------------
#! External data
#! --------------

path1 = r'try.png'

img = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)

plt.imshow(img,cmap='gray')
plt.title('Test Image')
plt.axis('off')
plt.show()
print('')

img = cv2.resize(img, (28, 28))
innn = torch.Tensor([i for i in img])
innn = innn.to(device)


#! --------------
#! Model
#! --------------

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        
        self.fc1=nn.Linear(32*4*4, 170, bias=True)
        self.fc2=nn.Linear(170, 56, bias=True)
        self.fc3=nn.Linear(56, 10, bias=True)
    
    def forward(self,x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        
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

    net_class.load_state_dict(torch.load(r'NNMnist.pt'))
    net_class = net_class.to(device)
    print('Model loaded\n')
    create = False
    
else:
    
    print('Creating a model\n')
    net_class=net_class.to(device)
    create = True
    
if create:
    
#! Data
    train=datasets.MNIST(r'',train=True,download=True,transform=transforms.Compose([
    transforms.ToTensor()
    ]))

    test=datasets.MNIST(r'',train=False,download=True,transform=transforms.Compose([
        transforms.ToTensor()
    ]))

    trainset= torch.utils.data.DataLoader(train,batch_size=10,shuffle=True)
    testset= torch.utils.data.DataLoader(test,batch_size=10,shuffle=True)

#! Training loop
    optimizer = optim.Adam(net_class.parameters(),lr= 0.0001)
    loss_function = nn.CrossEntropyLoss()

    epochs=3
    loss_grap=[]
    mean_grap=[]

    for epoch in range(epochs):
        loss_mean=0
        for data in tqdm(trainset, desc='Loss calculation'):

            x,y=data

            x,y = x.to(device), y.to(device)

            net_class.zero_grad()

            output=net_class(x.view(-1,1,28,28))

            loss = loss_function(output, y)

            loss.backward()

            optimizer.step()

            loss_grap.append(loss)

            loss_mean+=loss

        print(f'Epoch {epoch+1} error mean {(((loss_mean/len(trainset)).item())):.5f}')
        mean_grap.append(((loss_mean/len(trainset)).item()))

#! Plotting
    xpt=np.linspace(0,len(loss_grap),len(mean_grap))
    xpt=[int(i) for i in xpt]

    plt.plot(xpt,mean_grap,c='red',lw=3)

    plt.title('Error')
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()
    
#! Accuracy
    correct=0
    total=0

    with torch.no_grad():
        for data in tqdm(testset, desc='Accuracy calculation'):
            x,y=data
            x,y = x.to(device), y.to(device)
            output=net_class(x.view(-1,1,28,28))
            for idx,i in enumerate(output):
                if torch.argmax(i)==y[idx]:
                    correct +=1
                total+=1

    print('Accuracy: ',round(correct/total,3))
    
#! --------------
#! Testing
#! --------------

net_res=net_class(innn.view(-1,1,28,28))
_, indices = torch.max(net_res.data, 1)
plt.imshow(img,cmap='gray')
plt.title(f'Number {indices.item()}')
plt.axis('off')
plt.show()