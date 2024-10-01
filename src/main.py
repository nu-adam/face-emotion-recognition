from utils.video_utils import read_video
from utils.preprocessing import detect_faces, crop_faces, preprocess_face
from inference import inference

import torch


def main(video_path, model_path):
    """
    Main function for processing a video for emotion recognition.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the trained model for inference.
    """
    frames, fps = read_video(video_path)
    frames_per_second = int(fps)

    batch_tensors = []

    for frame_count, frame in frames:
        if frame_count % frames_per_second == 0:
            boxes = detect_faces(frame)
            if boxes is not None:
                face_crop = crop_faces(frame, boxes[0])
                face_tensor = preprocess_face(face_crop)
                batch_tensors.append(face_tensor)
    
    batch_tensors = torch.stack(batch_tensors)

    output = inference(batch_tensors, model_path)
    print(output)


if __name__ == '__main__':
    VIDEO_PATH = "videos/video_2.mp4"
    MODEL_PATH = "checkpoints/"

    main(VIDEO_PATH, MODEL_PATH)
