import torch
import torch.optim as optim
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
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, input):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) # flatten all dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

## Loss function and optimizer for when we get there
#criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

chocolate_dataset_train = datasets.ImageFolder('data')

dataset_tuple = chocolate_dataset_train.imgs
chocolate_dataset_train.class_to_idx

clss_vals = list(chocolate_dataset_train.class_to_idx.keys())
idx_vals = list(chocolate_dataset_train.class_to_idx.values())

data = []
counter = {}

for img_path in dataset_tuple:
    _,idx = img_path
    position = idx_vals.index(idx)
    counter[clss_vals[position]] = counter.get(clss_vals[position], 0) + 1
    data.append([clss_vals[position], _])

df = pd.DataFrame(data, columns = ['Class', 'File Path'])

figure = plt.figure(figsize=(8,8))
cols, rows = 3,3

for i in range(1, cols * rows + 1):
   sample_idx = torch.randint(len(dataset_tuple), size=(1,)).item()
   label,img = data[sample_idx]
   figure.add_subplot(rows, cols, i)
   plt.title("Labels")
   plt.axis("off")
#   plt.imshow(img.squeeze(), cmap="gray")
