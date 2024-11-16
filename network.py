import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
import os

chocolate_dataset_train = datasets.ImageFolder('data')

dataset_tuple = chocolate_dataset_train.imgs
chocolate_dataset_train.class_to_idx

clss_vals = list(chocolate_dataset_train.class_to_idx.keys())
idx_vals = list(chocolate_dataset_train.class_to_idx.values())

img_idx = {}


#Building pandas dataframe
#img_idx.fromkeys(list(clss_vals))

#print(img_idx)

df = pd.DataFrame(img_idx)

#print(df)

data = []

for img_path in dataset_tuple:
    _,idx = img_path
    position = idx_vals.index(idx)
    print(clss_vals[position], _)

### ultimately dataframe should look like clss, img as cols..each data in row formatted 


