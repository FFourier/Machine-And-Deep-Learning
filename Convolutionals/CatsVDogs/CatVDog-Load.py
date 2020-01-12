import os
import sys
import cv2 

import torch
import torch.nn as nn
import torch.nn.functional as F


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



files = [f for f in os.listdir('.') if os.path.isfile(f)]
saves = []
loaded = False

for f in files:
    if (f.find('.pt') != -1):
        saves.append(f)

if (len(saves))>0:
    
    index = int(input(f'What file do you want to load:\n\n\t{saves}\nSelect index: '))
    
    try:
        yn = input(f'\nLoad: {saves[index]}\n[y|n] ')

        if yn == 'y':
            net_class = Net()
            
            try:
                
                net_class.load_state_dict(torch.load(saves[index]))
                device = torch.device("cuda:0")
                net_class = net_class.to(device)
                print(f'\nNetwork loaded with {saves[index]}')
                loaded = True
                
            except AttributeError:
                print(f'\nSelected file is not usable, select a state dict type.')
        else:
            print(f'File not loaded')
                
    except IndexError:
        print(f'Index {index} is out of range, select index from 0 to {len(saves)-1} ')

if loaded:
    path = os.getcwd() + '\\imgs'
    files = [f for f in os.listdir(path)]
    
    d_files={}
    for idx,data in enumerate(files):
        d_files.update({idx : data})  
    index = int(input(f'\nImages avalibles in directory:\n\n{d_files}\n\nSelect index: '))
    
    try:
        
        print(f'\nImage {d_files[index]} was selected')
        f_path = path + '\\' + d_files[index]
        img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (50, 50))
        img = torch.Tensor([i for i in img])
        img = img.to(device)
        net_res = net_class(img.view(-1,1,50,50))
        print(f'\nThere is a {((net_res[0][0])*100):.2f} % certainty that it is a cat and a {((net_res[0][1])*100):.2f} % that it is a dog')
        
    except KeyError:
        print(f'Index {index} is out of range, select one from 0 to {len(d_files)-1}')