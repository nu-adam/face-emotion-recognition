import torch
from utils.model_utils import initialize_resnet18

def recognize_emotions(face_crops):
    """
    Recognizes emotions for a list of face crops.

    Args:
    - face_crops (list): List of preprocessed face tensors.

    Returns:
    - labels (list): List of emotion labels with probabilities.
    """
    model = initialize_resnet18()
    labels = []
    
    with torch.no_grad():
        for face_tensor in face_crops:
            logits = model(face_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1)
            max_prob, predicted_class = torch.max(probs, 1)
            emotion_classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            label = f"{emotion_classes[predicted_class]}: {max_prob.item():.2f}"
            labels.append(label)
    return labels
