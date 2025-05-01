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

def plot_confusion_matrices(y_true, y_pred_base, y_pred_trained, classes):
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot confusion matrix for base model
    cm1 = confusion_matrix(y_true, y_pred_base)
    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes, ax=ax1)
    ax1.set_title('Base Model Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Plot confusion matrix for trained model
    cm2 = confusion_matrix(y_true, y_pred_trained)
    sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes, ax=ax2)
    ax2.set_title('Fine-tuned Model Confusion Matrix')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
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
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Don't shuffle for consistent comparison
    
    # Initialize base model (untrained)
    base_model = PoopClassifier(num_classes=7)
    base_model = base_model.to(device)
    
    # Initialize and load trained model
    trained_model = PoopClassifier(num_classes=7)
    trained_model.load_state_dict(torch.load('best_model.pth'))
    trained_model = trained_model.to(device)
    
    # Evaluate both models
    print("Evaluating base model...")
    base_preds, true_labels = evaluate_model(base_model, dataloader, device)
    
    print("\nEvaluating trained model...")
    trained_preds, _ = evaluate_model(trained_model, dataloader, device)
    
    # Print classification reports
    class_names = [f"Class {i}" for i in range(1, 8)]
    
    print("\nBase Model Classification Report:")
    print(classification_report(true_labels, base_preds, target_names=class_names))
    
    print("\nTrained Model Classification Report:")
    print(classification_report(true_labels, trained_preds, target_names=class_names))
    
    # Plot confusion matrices
    plot_confusion_matrices(true_labels, base_preds, trained_preds, class_names)
    print("\nConfusion matrices have been saved as 'model_comparison.png'")

if __name__ == "__main__":
    main()
