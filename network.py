import torch
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) # flatten all dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

batch_size = 4
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((256,256)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

chocolate_dataset_train = datasets.ImageFolder('data', transform=transform)
chocolate_dataset_loader = torch.utils.data.DataLoader( chocolate_dataset_train, batch_size=batch_size, 
                                                       shuffle=True)

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

## pandas dataframe of dataset class, file_path
df = pd.DataFrame(data, columns = ['Class', 'File Path'])

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()

dataiter = iter(chocolate_dataset_loader)
images,labels = next(dataiter)

## print labels first
print(' '.join(f'{clss_vals[labels[j]]:5s}' for j in range(batch_size)))
imshow(torchvision.utils.make_grid(images))

## Defining a loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

## Train the network
# for epoch in range(2):

#     running_loss = 0.0
#     for i, data in enumerate(chocolate_dataset_loader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data
#         # zero the parameter gradients
#         optimizer.zero_grad()

#         #forward + backward + optimizer
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         ## print statistics
#         running_loss += loss.item()
#         if i % 10 == 9:
#             print('[%d, %5d] loss: %.3f' %
#                  ( epoch + 1, i + 1, running_loss/ 10  ))
#             running_loss = 0.0
# print('Finished Training')
