import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from dataset import PoopDataset, PoopClassifier
from torch.utils.data import DataLoader
import torch.nn as nn

class FeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.features = {}
        self.hooks = []
        
        # Register hooks for different layers
        self._register_hooks()
    
    def _register_hooks(self):
        # Initial layers
        self.hooks.append(
            self.model.base_model.conv1.register_forward_hook(
                lambda m, i, o: self._add_feature('conv1', o)
            )
        )
        
        # First layer of each residual block
        for idx, layer in enumerate(self.model.base_model.layer1):
            self.hooks.append(
                layer.conv1.register_forward_hook(
                    lambda m, i, o, idx=idx: self._add_feature(f'layer1_{idx}_conv1', o)
                )
            )
        
        for idx, layer in enumerate(self.model.base_model.layer2):
            self.hooks.append(
                layer.conv1.register_forward_hook(
                    lambda m, i, o, idx=idx: self._add_feature(f'layer2_{idx}_conv1', o)
                )
            )
    
    def _add_feature(self, name, output):
        # Store the output of the layer
        self.features[name] = output.detach()
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

def visualize_feature_maps(feature_maps, layer_name, num_features=8):
    # Get the first image's feature maps
    features = feature_maps[0].cpu()  # Get first image in batch
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_features)))
    
    plt.figure(figsize=(15, 15))
    plt.suptitle(f'Feature Maps for {layer_name}', size=16)
    
    for idx in range(min(num_features, features.shape[0])):
        plt.subplot(grid_size, grid_size, idx + 1)
        plt.imshow(features[idx], cmap='viridis')
        plt.axis('off')
        plt.title(f'Filter {idx}')
    
    plt.savefig(f'feature_maps_{layer_name}.png')
    plt.close()

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loading
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset and get a single batch
    dataset = PoopDataset("data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Get a single image
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    # Load model
    model = PoopClassifier(num_classes=7)
    model = model.to(device)
    model.eval()
    
    # Create feature extractor
    feature_extractor = FeatureExtractor(model)
    
    # Forward pass
    with torch.no_grad():
        _ = model(images)
    
    # Visualize features for each layer
    for layer_name, feature_maps in feature_extractor.features.items():
        print(f"Visualizing {layer_name}")
        visualize_feature_maps(feature_maps, layer_name)
    
    # Clean up
    feature_extractor.remove_hooks()
    
    print("\nVisualization complete! Check the current directory for feature map images.")
    
    # Save the original image for reference
    plt.figure(figsize=(5, 5))
    img = images[0].cpu().permute(1, 2, 0)
    # Denormalize
    img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
    img = torch.clamp(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Original Image')
    plt.savefig('original_image.png')
    plt.close()

if __name__ == "__main__":
    main()
