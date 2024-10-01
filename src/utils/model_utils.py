from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train_model(train_loader, model, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Trains the Face Emotion Recognition model on the training data using transfer learning.
    
    Args:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - model (nn.Module): Face Emotion Recognition model for training.
    - criterion (nn.Module): Loss function for training.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - num_epochs (int): Number of epochs to train the model.
    - device (str): Device to use for training ('cuda' or 'cpu').

    Returns:
    - None
    """
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0

        for inputs, labels in tqdm(train_loader, unit="batch", desc=f"Epoch [{epoch+1}/{num_epochs}]", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        checkpoint_path = f'checkpoints/model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')


def evaluate_model(model, data_loader, device):
    """
    Evaluates the performance of the Face Emotion Recognition model on the test dataset.

    Args:
    - model (torch.nn.Module): The trained Face Emotion Recognition model.
    - data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - device (torch.device): Device to perform computations on ('cuda' or 'cpu').

    Returns:
    - dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, unit="batch", desc="Evaluating", ncols=100):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }
