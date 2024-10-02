import torch
import torch.optim as optim
import torch.nn as nn
import os

from utils.data_utils import load_data
from utils.model_utils import train_model, evaluate_model
from utils.models import FaceEmotionModel


def train(data_dir, num_classes, batch_size=32, learning_rate=0.001, num_epochs=10, checkpoint_dir='results/checkpoints/'):
    """
    Training for the Face Emotion Recognition model using transfer learning on the specified dataset.

    Args:
    - data_dir (str): Path to the root directory containing 'train' and 'test' subdirectories.
    - num_classes (int): Number of emotion classes for classification.
    - batch_size (int, optional): Number of samples per batch to load. Default is 32.
    - learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
    - num_epochs (int, optional): Number of epochs to train the model. Default is 10.
    - checkpoint_dir (str, optional): Directory to save model checkpoints. Default is 'results/checkpoints/'.

    Returns:
    - None
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the dataset
    train_loader, val_loader, test_loader = load_data(data_dir, batch_size=batch_size)

    # Initialize the model
    model = FaceEmotionModel(embed_dim=512, num_heads=4, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs, device, checkpoint_dir)

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")
    print(f"Test F1 Score: {metrics['f1_score']:.4f}")


if __name__ == '__main__':
    # Configuration parameters
    DATA_DIR = r'C:\dev\face-emotion-recognition\dataset'
    NUM_CLASSES = 7
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 1
    CHECKPOINT_DIR = 'results/checkpoints/'

    train(DATA_DIR, NUM_CLASSES, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS, CHECKPOINT_DIR)
