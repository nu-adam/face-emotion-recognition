import torch
from utils.models import FaceEmotionModel


def inference(batch_tensors, model_path, device='cpu'):
    """
    Recognizes emotions for a list of face crops.

    Args:
    - batch_tensors (list): List of preprocessed face tensors.

    Returns:
    - labels (list): List of emotion labels with probabilities.
    """

    emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    model = FaceEmotionModel(embed_dim=512, num_heads=4, num_layers=2)
    model.to(device)
    model.eval()
    
    labels = []

    with torch.no_grad():
        logits = model(batch_tensors)
        probs = torch.nn.functional.softmax(logits, dim=1)
        max_prob, predicted_class = torch.max(probs, 1)
        for prob, class_ in zip(max_prob, predicted_class):
            label = f"{emotion_classes[class_]}: {prob.item():.5f}"
            labels.append(label)
        
    return labels
