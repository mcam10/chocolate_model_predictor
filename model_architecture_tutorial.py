import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import functional as F

class ModelArchitectureTutorial:
    """Tutorial class demonstrating different ways to modify model architectures"""
    
    @staticmethod
    def inspect_resnet_structure():
        """Inspect the layer structure of ResNet50"""
        model = resnet50()
        print("=== ResNet50 Structure ===")
        print("\nMain components:")
        print("1. Backbone (Feature Extractor):")
        print("   - conv1: Initial convolution")
        print("   - bn1: Batch normalization")
        print("   - relu: Activation")
        print("   - maxpool: Max pooling")
        print("   - layer1-4: Residual blocks")
        print("\n2. Head (Classifier):")
        print("   - avgpool: Global average pooling")
        print("   - fc: Final fully connected layer")
        
        # Print detailed layer information
        print("\nDetailed layer sizes:")
        sample_input = torch.randn(1, 3, 224, 224)
        activation = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.shape
            return hook
        
        # Register hooks
        model.conv1.register_forward_hook(get_activation('conv1'))
        model.layer1.register_forward_hook(get_activation('layer1'))
        model.layer2.register_forward_hook(get_activation('layer2'))
        model.layer3.register_forward_hook(get_activation('layer3'))
        model.layer4.register_forward_hook(get_activation('layer4'))
        model.fc.register_forward_hook(get_activation('fc'))
        
        # Forward pass
        with torch.no_grad():
            model(sample_input)
        
        # Print activation shapes
        for name, shape in activation.items():
            print(f"{name}: {shape}")

    @staticmethod
    def create_custom_backbone():
        """Example of creating a custom backbone"""
        class CustomBackbone(nn.Module):
            def __init__(self, in_channels=3):
                super().__init__()
                # Simple backbone example
                self.features = nn.Sequential(
                    # First conv block
                    nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    
                    # Second conv block
                    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    
                    # Third conv block
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                )
            
            def forward(self, x):
                return self.features(x)
        
        return CustomBackbone()

    @staticmethod
    def create_custom_head(in_features, num_classes):
        """Example of creating a custom classification head"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    @staticmethod
    def modify_existing_model():
        """Example of modifying an existing model"""
        # Load pre-trained ResNet
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 1. Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
            
        # 2. Replace classification head
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 7)  # 7 classes for our case
        )
        
        return model

    @staticmethod
    def create_feature_pyramid():
        """Example of creating a Feature Pyramid Network (FPN)"""
        class FeaturePyramid(nn.Module):
            def __init__(self):
                super().__init__()
                # Bottom-up pathway (backbone)
                resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
                self.layer1 = nn.Sequential(
                    resnet.conv1,
                    resnet.bn1,
                    resnet.relu,
                    resnet.maxpool,
                    resnet.layer1
                )
                self.layer2 = resnet.layer2
                self.layer3 = resnet.layer3
                self.layer4 = resnet.layer4
                
                # Top-down pathway (lateral connections)
                self.lateral4 = nn.Conv2d(2048, 256, 1)
                self.lateral3 = nn.Conv2d(1024, 256, 1)
                self.lateral2 = nn.Conv2d(512, 256, 1)
                self.lateral1 = nn.Conv2d(256, 256, 1)
                
            def forward(self, x):
                # Bottom-up
                c1 = self.layer1(x)
                c2 = self.layer2(c1)
                c3 = self.layer3(c2)
                c4 = self.layer4(c3)
                
                # Top-down
                p4 = self.lateral4(c4)
                p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:])
                p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:])
                p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:])
                
                return [p1, p2, p3, p4]
        
        return FeaturePyramid()

def main():
    tutorial = ModelArchitectureTutorial()
    
    print("\n=== 1. Understanding ResNet Structure ===")
    tutorial.inspect_resnet_structure()
    
    print("\n=== 2. Custom Backbone Example ===")
    backbone = tutorial.create_custom_backbone()
    print(backbone)
    
    print("\n=== 3. Custom Head Example ===")
    head = tutorial.create_custom_head(256, 7)
    print(head)
    
    print("\n=== 4. Modified ResNet Example ===")
    modified_model = tutorial.modify_existing_model()
    print(modified_model.fc)  # Show modified head
    
    print("\n=== 5. Feature Pyramid Example ===")
    fpn = tutorial.create_feature_pyramid()
    print("FPN created with multi-scale feature extraction")

if __name__ == "__main__":
    main()
