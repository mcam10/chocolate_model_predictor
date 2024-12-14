import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib as plt

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 sqaure convs
        self.conv1 = nn.Conv2d(1,6,5)
    def forward(self, input):
        ## Convolution layer C!: 1 input image channel, 6 output channels, 
        c1 = F.relu(self.conv1(input))

net = Net()
print(net)

params = list(net.parameters())
print(params[0].size())
