from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def initialize_mtcnn():
    """
    Initializes the MTCNN model for face detection.

    Returns:
    - mtcnn (MTCNN): An initialized MTCNN model.
    """
    mtcnn = MTCNN(keep_all=True)
    return mtcnn


def initialize_resnet18():
    """
    Initializes a pretrained ResNet18 model for emotion recognition.

    Returns:
    - model (torch.nn.Module): Pretrained emotion recognition model.
    """
    model = models.resnet18(pretrained=True)
    
    # Modify the final layer to match the number of emotion classes
    num_classes = 7
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # model.load_state_dict(torch.load('models/pretrained_weights.pth'))
    model.eval()

    return model


def initialize_vgg19(num_classes=7):
    """
    Initializes a pretrained VGG19 model with transfer learning for emotion recognition.
    
    Loads a pre-trained VGG19 model, freezes its convolutional layers, and modifies
    the classifier to match the number of emotion classes.

    Args:
    - num_classes (int): The number of output classes for emotion recognition.

    Returns:
    - model (nn.Module): The modified VGG19 model.
    """
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    
    # Freeze the convolutional layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace the classifier to match the number of classes
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    
    return model


def train_vgg19(train_loader, model, criterion, optimizer, num_epochs=10, device='cuda'):
    """
    Trains the VGG19 model on the training data using transfer learning.
    
    Args:
    - train_loader (DataLoader): DataLoader for the training dataset.
    - model (nn.Module): VGG19 model for training.
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
        with tqdm(train_loader, unit="batch", desc=f"Epoch [{epoch+1}/{num_epochs}]", ncols=100) as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)

                pbar.set_postfix(loss=loss.item())
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
        
        checkpoint_path = f'checkpoints/vgg19/vgg19_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), checkpoint_path)
        print(f'Saved checkpoint: {checkpoint_path}')


def evaluate_vgg19(model, data_loader, device):
    """
    Evaluates the performance of the VGG19 model on the test dataset.

    Args:
    - model (torch.nn.Module): The trained VGG19 model.
    - data_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - device (torch.device): Device to perform computations on ('cuda' or 'cpu').

    Returns:
    - dict: A dictionary containing accuracy, precision, recall, and F1-score.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        with tqdm(data_loader, unit="batch", desc="Evaluating", ncols=100) as pbar:
            for images, labels in data_loader:
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
