import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 sqaure convs
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        # an affline operation y = mx+b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, input):
        # Convolution layer C1: 1 input image channel, 6 output channels,
        # 5x5 square convolution, it uses RELU activation function, and
        # outputs a Tensor with size (N, 6, 28, 28), where N is the size of the batch
        c1 = F.relu(self.conv1(input))

        s2 = F.max_pool2d(c1, (2,2))

        c3 = F.relu(self.conv2(s3))

        s4 = F.max_pool2d(c3,2)

        s4 = torch.flatten(s4,2)

        f5 = F.relu(self.fc1(s4))

        f6 = F.relu(self.fc1(s5))

        output = self.fc3(f6)
        return output

net = Net()
## printing network information
print('Network......')
print(net)
params = list(net.parameters())
print('Params......')
print(params)
#print(params[0].size())

## Get the first convulutional layer
conv_layer = net.conv1

print('Conv1 layer......')
print(conv_layer)

## Get the weights of the layer
filters = conv_layer.weight.data.cpu().numpy()

print('Filters......')
print(filters)

## Lets plot the filters

#fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(8,8))

#for i,ax in enumerate(axes.flat):
#    ax.imshow(filters[:, :, :, i], cmap='gray')
#    ax.axis('off')

#plt.show

