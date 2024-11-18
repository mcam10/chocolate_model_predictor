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

df = pd.DataFrame()

data = []

for img_path in dataset_tuple:
    _,idx = img_path
    position = idx_vals.index(idx)
    data.append([clss_vals[position], _])

df = pd.DataFrame(data, columns = ['Class', 'File Path'])

print(df)


