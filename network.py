import torch
from torch.utils.data import Dataset
from torchvision import datasets

chocolate_dataset = datasets.ImageFolder('data')

print(chocolate_dataset)


