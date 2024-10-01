import torch
import torch.nn as nn
from torchvision import models


class VGGFeatureExtractor(nn.Module):
    """
    Feature Extractor based on the VGG-19 architecture.
    """
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        self.vgg = models.vgg19(weights=None).features

    def forward(self, x):
        features = self.vgg(x)
        return features  # Output: (batch_size, 512, 7, 7)
    

class TransformerEncoder(nn.Module):
    """
    Transformer model for processing image features.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2):
        super(TransformerEncoder, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        
    def forward(self, x):
        x = self.transformer(x)
        return x  # Output: (batch_size, 49, embed_dim)
    

class ProjectionNetwork(nn.Module):
    """
    A projection network to reduce the dimensionality of the VGG feature maps.
    """
    def __init__(self, input_dim=512, output_dim=256):
        super(ProjectionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, output_dim)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return x  # Output: (batch_size, 49, output_dim)


class FaceEmotionModel(nn.Module):
    """
    Face Emotion Recognition Model combining VGG-19 and Transformer.

    This model extracts features from facial images using VGG19, then processes
    these features through a Transformer to learn temporal and spatial
    representations, and finally classifies the emotions present in the input.
    """
    def __init__(self, embed_dim=256, num_heads=4, num_layers=2, num_classes=7):
        super(FaceEmotionModel, self).__init__()
        self.feature_extractor = VGGFeatureExtractor()
        self.transformer = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.projection = ProjectionNetwork(input_dim=512, output_dim=embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, 49, embed_dim))
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 512, 7 * 7).permute(0, 2, 1) # Shape: (batch_size, 49, 512)
        x = self.projection(x) # Shape: (batch_size, 49, embed_dim)
        x = x + self.positional_encoding
        x = self.transformer(x)
        x = x.mean(dim=1) # Shape: (batch_size, embed_dim)
        x = self.classifier(x) # Shape: (batch_size, num_classes)
        return x # Output: (batch_size, num_classes)
