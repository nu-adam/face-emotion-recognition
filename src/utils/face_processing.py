import torch
from torchvision import transforms
from PIL import Image


def preprocess_face(face_crop, input_size=(224, 224)):
    """
    Preprocesses a single face crop for the emotion recognition model.

    Args:
    - face_crop (numpy.ndarray): The cropped face image in RGB format.

    Returns:
    - torch.Tensor: Preprocessed face tensor ready for model input.
    """
    preprocess_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    face_tensor = preprocess_transform(face_crop)
    
    return face_tensor.unsqueeze(0)


def batch_process_faces(face_crops):
    """
    Processes a batch of face crops for the emotion recognition model.

    Args:
    - face_crops (list of numpy.ndarray): List of cropped face images.

    Returns:
    - torch.Tensor: Batch of preprocessed face tensors.
    """
    face_tensors = [preprocess_face(face_crop) for face_crop in face_crops]
    
    batch_tensor = torch.stack(face_tensors)
    
    return batch_tensor
