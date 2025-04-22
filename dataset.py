from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch

class PoopDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = datasets.ImageFolder(data_dir, transform=transform)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

#Defining our transformations to pass into our dataset    
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Resize to standard size that resnet expects
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

dataset = PoopDataset(data_dir="data", transform=transform)

## Confirmation for correctness of data
#print(dataset[67])
# print(len(dataset))
# print(dataset.classes)
#image,label = dataset[67]
#print(image.shape) ## RGB channel x SIZE x SIZE

## Pull a random 32 images everytime we load from dataset. 
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    break
# print(image.shape) ## batch size, RGB channel x SIZE x SIZE

class PoopClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        # Initialize with modern weights
        weights = ResNet50_Weights.DEFAULT
        self.base_model = resnet50(weights=weights)

        # Freeze the feature extractor
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the classifier
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.base_model(x)