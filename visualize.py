import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
import os

import matplotlib.pyplot as plt

chocolate_dataset_train = datasets.ImageFolder('data')

dataset_tuple = chocolate_dataset_train.imgs
chocolate_dataset_train.class_to_idx

clss_vals = list(chocolate_dataset_train.class_to_idx.keys())
idx_vals = list(chocolate_dataset_train.class_to_idx.values())

df = pd.DataFrame()

data = []
counter = {}


for img_path in dataset_tuple:
    _,idx = img_path
#    counter[idx] = counter.get(idx,0) + 1
    position = idx_vals.index(idx)
    counter[clss_vals[position]] = counter.get(clss_vals[position], 0) + 1
    data.append([clss_vals[position], _])

df = pd.DataFrame(data, columns = ['Class', 'File Path'])

## Build some visualizations
plt.bar(counter.keys(), counter.values())
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Bar Chart of Counts of each sample')

####
plt.show()



