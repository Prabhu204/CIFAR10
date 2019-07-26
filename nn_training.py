# -*- coding: utf-8 -*-
"""
@author: Prabhu <prabhu.appalapuri@gmail.com>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam



class Simple_nn(nn.Module):
    def __init__(self):
        super(Simple_nn, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 6, 3), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 2),nn.ReLU(), nn.MaxPool2d(2))
        print(self.conv1)
        self.fc1 = nn.Linear(16*7*7, 120)
        self.fc2  = nn.Linear(120, 10)
        print(self.conv2)

    def forward(self, x):
        output = self.conv1(x)
        output= self.conv2(output)
        print(output.size())
        output = output.view(-1,self.num_faltfeatures(output))
        output = self.fc1(output)
        out = self.fc2(output)
        return out


    def num_faltfeatures(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *=s
        return num_features

model = Simple_nn()
print(model)
input = torch.randn(1,1,32,32)
print(input)
torch.manual_seed(14)
target = torch.randn(10)
target = target.view(1, -1)
print(target)
opti = Adam(model.parameters(),lr=0.001)
for i in range(4):
    opti.zero_grad()
    output = model(input)
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    opti.step()


