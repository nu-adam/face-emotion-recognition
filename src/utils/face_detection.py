from utils.model_utils import initialize_mtcnn


def detect_faces(frame):
    """
    Detect faces in a given frame using MTCNN.

    Args:
    - frame (numpy array): The input frame in RGB format.

    Returns:
    - boxes (list): List of bounding boxes for detected faces, each in (x1, y1, x2, y2) format.
    """
    model = initialize_mtcnn()
    boxes, _ = model.detect(frame)
    return boxes if boxes is not None else []


def crop_faces(frame, boxes):
    """
    Crops faces from the frame based on the provided bounding boxes.

    Args:
    - frame (numpy array): The input frame in RGB format.
    - boxes (list): List of bounding boxes for detected faces.

    Returns:
    - face_crops (list of numpy arrays): List of cropped face images.
    """
    face_crops = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        face_crop = frame[y1:y2, x1:x2]
        face_crops.append(face_crop)
    return face_crops
