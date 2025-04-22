import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import PoopDataset, PoopClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    dataset = PoopDataset("data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = PoopClassifier(num_classes=7)
    model = model.to(device)

    # Evaluate
    print("Evaluating model...")
    predictions, true_labels = evaluate_model(model, dataloader, device)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions, 
                              target_names=[f"Class {i}" for i in range(1, 8)]))
    
    # Plot confusion matrix
    plot_confusion_matrix(true_labels, predictions, 
                         classes=[f"Class {i}" for i in range(1, 8)])
    print("\nConfusion matrix has been saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()
