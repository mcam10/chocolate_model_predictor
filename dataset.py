import torch
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

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
    transforms.Resize((224, 224)), # Resize to standard size
    transforms.ToTensor()
])

dataset = PoopDataset(data_dir="data", transform=transform)

## Confirmation for correctness of datq
#print(dataset[67])
# print(len(dataset))
# print(dataset.classes)
#image,label = dataset[67]
image, label = dataset[67]
#print(image.shape) ## RGB channel x SIZE x SIZE

## Pull a random 32 images everytime we load from dataset. 
dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

for images, labels in dataloader:
    break

#print(image.shape) ## batch size, RGB channel x SIZE x SIZE
print(images.shape)