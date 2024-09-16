from facenet_pytorch import MTCNN
import torch
import torch.nn as nn
from torchvision import models


def initialize_mtcnn():
    """
    Initializes the MTCNN model with the specified device.

    Returns:
    - mtcnn (MTCNN): An initialized MTCNN model.
    """
    mtcnn = MTCNN(keep_all=True)
    return mtcnn


def initialize_resnet18():
    """
    Initializes and loads the pretrained emotion recognition model.

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
