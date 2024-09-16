import cv2


def draw_face_boxes(frame, boxes):
    """
    Draws bounding boxes on the frame for each detected face.

    Args:
    - frame (numpy array): The input frame.
    - boxes (list of tuples): List of bounding box coordinates [(x1, y1, x2, y2), ...].

    Returns:
    - frame (numpy array): Frame with bounding boxes drawn.
    """
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame


def add_emotion_labels(frame, boxes, labels):
    """
    Adds emotion labels with probabilities to each face box on the frame.

    Args:
    - frame (numpy array): The input frame.
    - boxes (list of tuples): List of bounding box coordinates [(x1, y1, x2, y2), ...].
    - labels (list of str): List of emotion labels to be added.

    Returns:
    - frame (numpy array): Frame with emotion labels added.
    """
    for box, label in zip(boxes, labels):
        x1, y1, _, _ = map(int, box)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame
