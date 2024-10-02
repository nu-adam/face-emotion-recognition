import os
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def save_model(state, checkpoint_dir='results/checkpoints'):
    """
    Saves the model state to a checkpoint file.

    Args:
    - state (dict): Dictionary containing model state, optimizer state, and other metadata.
    - checkpoint_dir (str): Directory where the checkpoints are saved.
    """
    filename = f'{checkpoint_dir}/best_model.pth'
    torch.save(state, filename)


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Trains the model on the train dataset for one epoch.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - criterion (torch.nn.Module): Loss function used for training.
    - optimizer (torch.optim.Optimizer): Optimizer to update model weights.
    - device (torch.device): Device to use for training ('cuda' or 'cpu').

    Returns:
    - epoch_loss (float): Average training loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, labels in tqdm(train_loader, unit="batch", desc="Training", ncols=100):
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
    return epoch_loss


def validate_one_epoch(model, val_loader, criterion, device):
    """
    Validates the model on the validation dataset for one epoch.

    Args:
    - model (torch.nn.Module): The model to be validated.
    - val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - criterion (torch.nn.Module): Loss function used for validation.
    - device (torch.device): Device to use for validation ('cuda' or 'cpu').

    Returns:
    - tuple: (float) Average validation loss over the epoch, (float) Validation accuracy.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, unit="batch", desc="Validation", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    accuracy = correct / total

    return epoch_loss, accuracy


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10, device='cuda', checkpoint_dir='results/checkpoints'):
    """
    Trains and validates the Face Emotion Recognition model using specified model.
    
    Args:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - val_loader (DataLoader): DataLoader for the validation dataset.
    - model (nn.Module): Face Emotion Recognition model for training.
    - criterion (nn.Module): Loss function for training.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - num_epochs (int): Number of epochs to train the model.
    - device (str): Device to use for training ('cuda' or 'cpu').
    - checkpoint_dir (str): Directory to save the model checkpoints.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    best_loss = float('inf')

    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate after one epoch
        val_loss, val_accuracy = validate_one_epoch(model, val_loader, criterion, device)

        # Save the best model checkpoint
        if val_loss  < best_loss:
            best_loss = val_loss 
            checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
            }
            save_model(checkpoint, checkpoint_dir=checkpoint_dir)


def compute_metrics(all_labels, all_preds):
    """
    Computes accuracy, precision, recall, and F1-score from predictions and ground truth labels.

    Args:
    - all_labels (list or np.array): True labels.
    - all_preds (list or np.array): Model predictions.

    Returns:
    - dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

    return metrics


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
        for images, labels in tqdm(data_loader, unit="batch", desc="Evaluation", ncols=100):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    metrics = compute_metrics(all_labels, all_preds)

    return metrics
