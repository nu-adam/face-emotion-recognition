from utils.video_utils import read_video, save_video
from utils.face_detection import detect_faces, crop_faces
from utils.visualization import draw_face_boxes, add_emotion_labels
from utils.face_processing import batch_process_faces
from utils.emotion_recognition import recognize_emotions

import cv2


def main():
    video_path = "videos/input_video.mp4"
    output_path = "videos/output_video.mp4"

    output_frames = []
    frames, fps = read_video(video_path)

    frames_per_second = int(fps)
    last_labels = []

    for frame_count, frame in frames:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = detect_faces(frame_rgb)

        if boxes is not None:
            if frame_count % frames_per_second == 0:
                labels = []
                face_crops = crop_faces(frame_rgb, boxes)
                face_tensors = batch_process_faces(face_crops)
                labels = recognize_emotions(face_tensors)
                last_labels = labels
            else:
                labels = last_labels
            
            frame = draw_face_boxes(frame, boxes)
            frame = add_emotion_labels(frame, boxes, labels)

        output_frames.append(frame)

    save_video(output_path, output_frames, fps)


if __name__ == '__main__':
    main()
