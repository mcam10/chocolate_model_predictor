import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
import os

chocolate_dataset_train = datasets.ImageFolder('data')

dataset_tuple = chocolate_dataset_train.imgs
print(chocolate_dataset_train.class_to_idx)

img_idx = {}

for img_path in dataset_tuple:
    _,idx = img_path


