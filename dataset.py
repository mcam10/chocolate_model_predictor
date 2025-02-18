from torchvision import datasets,transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])  # ImageNet normalization
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
# print(image.shape) ## batch size, RGB channel x SIZE x SIZE
# print(images.shape)

class PoopClassifier(nn.Module):
    def __init__(self, num_classes=7):
        super(PoopClassifier, self).__init__()
        ## where we all define parts of the model
        self.base_model = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        rnet_out_size = 2048 ## features from resnet
        self.classifier = nn.Linear(rnet_out_size, num_classes)

    def forward(self, x):
        # connect these parts and return the output
        x = self.features(x)
        output = self.classifier(x)
        return output

## Testing
pretrained_model_efficientnet = models.efficientnet_b0(pretrained=True)
#print(f"EffecientNet: {pretrained_model_efficientnet}")

pretrained_model_resnet = models.resnet50(pretrained=True)
#print(f"ResNet: {pretrained_model_resnet}")

model = PoopClassifier(num_classes=7)
## get model structure
#print(model)

#print(model(images))