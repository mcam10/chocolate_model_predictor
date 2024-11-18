import torch
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import pandas as pd
import os

chocolate_dataset_train = datasets.ImageFolder('data')
