import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import utils
from torchvision import datasets
from torchvision import transforms

from matplotlib import pyplot as plt

from tqdm import tqdm, trange

from collections import Counter

import cv2 





data_path = 'data'
train = datasets.CIFAR10(data_path, train=True, download=True, transform=transforms.Compose([
                            transforms.Grayscale(num_output_channels=1), 
                            transforms.ToTensor()]))

test = datasets.CIFAR10(data_path, train=False, download=True, transform=transforms.Compose([
                            transforms.Grayscale(num_output_channels=1), 
                            transforms.ToTensor()]))

classes = ['plane','car','bird','cat','deer','dog','frog','horse','ship','truck']


dc = [3,4,5,6,9]

t_label=[]
t_img=[]

for img, label in train:
    if label != dc[0] and label != dc[1] and label != dc[2] and label != dc[3] and label != dc[4]:
        t_img.append(img)
        t_label.append(label)
    

v_label=[]
v_img=[]

for img, label in test:
    if label != dc[0] and label != dc[1] and label != dc[2] and label != dc[3] and label != dc[4]:
        v_img.append(img)
        v_label.append(label)

        
erased=[]
for i in range(len(dc)):
    erased.append(classes.pop(dc[i]-i))
    
    
t_label = [3 if x == 8 else x for x in t_label]
t_label = [4 if x == 7 else x for x in t_label]

v_label = [3 if x == 8 else x for x in v_label]
v_label = [4 if x == 7 else x for x in v_label]


f_index=list(Counter(t_label).keys())
f_index.sort()

X_s = torch.Tensor(len(t_img), len(t_img[0]), len(t_img[0][0]), len(t_img[0][0][0]))
X_a_s = torch.Tensor(len(v_img), len(v_img[0]), len(v_img[0][0]), len(v_img[0][0][0]))

X = torch.cat(t_img, dim=0, out=X_s)
Y = torch.tensor(t_label)

X_a = torch.cat(v_img, dim=0, out=X_a_s)
Y_a = torch.tensor(v_label)






class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        
        self.fc1=nn.Linear(32*5*5, 170, bias=True)
        self.fc2=nn.Linear(170, 56, bias=True)
        self.fc3=nn.Linear(56, 5, bias=True)
    
    def forward(self,x):
        
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        
        x = torch.flatten(x, start_dim=1)
        
        x= F.relu(self.fc1(x)) 
        x= F.relu(self.fc2(x))
        x= self.fc3(x)
        
        return F.softmax(x, dim=1)
    

    
    
torch.cuda.empty_cache()
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    net_class = Net().to(device)
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    net_class = Net()
    print("Running on the CPU")
    
    
    

    
batch_size = 10
epochs = 35

optimizer = optim.Adam(net_class.parameters(),lr= 0.0008)
loss_function = nn.CrossEntropyLoss()

loss_grap=[]
mean_grap=[]
prin = True

for epoch in range(epochs):
    loss_mean=0
    for i in trange(0, len(X), batch_size,  desc='Loss calculation'):
        
        batch_X=X[i:i+batch_size].view(-1,1,32,32)
        batch_y=Y[i:i+batch_size]
        
        x,y = batch_X.to(device), batch_y.to(device)
        
        net_class.zero_grad()
        
        output=net_class(x.view(-1,1,32,32))
        
        if prin:
            print(f'\nOutput shape: {output.shape}\t Y shape: {y.shape}\n')
            prin = False
        
        loss = loss_function(output, y)
        
        loss.backward()
        
        optimizer.step()
        
        loss_grap.append(loss)
    
        loss_mean+=loss
        
    print(f'Epoch {epoch+1} error mean {((loss_mean/i).item()):.5f}')
    mean_grap.append((loss_mean/i).item())
    
    


    
    
xpt=np.linspace(0,len(loss_grap),len(mean_grap))
xpt=[int(i) for i in xpt]

plt.plot(xpt,mean_grap,c='red',lw=3)

plt.title('Error')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()






correct = 0
total = 0
with torch.no_grad():
    
    for i in trange(len(X_a), desc='Accuracy calculation'):
        
        X_a,Y_a = X_a.to(device), Y_a.to(device)
        
        net_out = net_class(X_a[i].view(-1,1,32,32))[0]
        
        predicted_class = torch.argmax(net_out)
        
        if predicted_class == Y_a[i]:
            correct += 1
        total += 1

print("Accuracy: ",round(correct/total,3))




#Path
path1 = r'CIFAR1\imgs\der.jpg'
path2 = r'CIFAR1\imgs\carr.jpg'
path3 = r'CIFAR1\imgs\shi.png'
path4 = r'CIFAR1\imgs\pig.jpg'
path5 = r'CIFAR1\imgs\plan.jpg'
path6 = r'CIFAR1\imgs\pgu.jpg'

#Image transformation
img1 = cv2.imread(path6, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img1, (32, 32))
innn = torch.Tensor([i for i in img])
innn = innn.to(device)

net_res = net_class(innn.view(-1,1,32,32))
p_class = torch.argmax(net_res)

#Plotting
plt.imshow(img1,cmap='gray')
plt.title(f_class[p_class.item()], fontsize=30)
plt.axis('off')
plt.show()




sv = False
if sv:
    torch.save(net_class.state_dict(), r'CIFAR1\NNCIFR10.pt')
    torch.save(net_class, r'CIFAR1\NNCifar-10.pt')
    print('Model saved')
else:
    print('Model not saved')