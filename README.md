# Face Emotion Recognition

## Overview

This repository contains the **face emotion recognition** implementation of a **multimodal emotion recognition system** based on the paper **"Multi-Label Multimodal Emotion Recognition With Transformer-Based Fusion and Emotion-Level Representation Learning"**. The current implementation focuses on **video processing** for emotion recognition from facial expressions, including face detection, feature extraction, and temporal sequence modeling.

### Key Components:
1. **Video Capturing**: Processing videos frame by frame.
2. **Face Detection**: Using MTCNN to detect and crop faces.
3. **Feature Extraction**: Extracting features from face crops using VGG19.
4. **Transformer-based Embeddings**: Passing extracted features through a Transformer to generate temporal emotion embeddings and predict the emotion.

---

## Table of Contents
1. [Installation](#installation)
2. [Model Architecture](#model-architecture)
3. [Usage](#usage)
   - [Training](#training)
   - [Inference](#inference)
4. [Future Plans](#future-plans)
5. [Contributing](#contributing)
6. [License](#license)

---

## Installation

To run this project locally, you'll need Python 3.7+ and some dependencies. Follow these steps:

1. Clone the repository:
```bash
   git clone https://github.com/nu-adam/face-emotion-recognition.git
   cd face-emotion-recognition
```

2. Set up a virtual environment:
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate.bat
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Model Architecture

The architecture consists of two main components:

1. **VGG19 for Feature Extraction**:
   - VGG19 model trained from scratch is used to extract 512-dimensional feature vectors from cropped faces in the video.
   - Each frame’s face is represented as a feature vector of shape `(512, 7, 7)`.

2. **Transformer for Temporal Modeling**:
   - The extracted features from VGG19 are reshaped and fed into a Transformer.
   - The Transformer captures the temporal relationships between frames and outputs classified emotion.

The overall architecture is designed to process face crops from each frame, extract spatial features, and then model temporal dependencies between the frames for emotion prediction.

---

## Usage

### Training

To train a model on a new dataset, follow these steps:

1. **Dataset Organization**:
   - Before training, ensure that the **FER-2013** dataset is correctly organized in the `dataset/` folder. The files should be stored in this folder, and the system will process them automatically.

2. **Set the Data Directory**:
   - Open the `train.py` file and modify the `DATA_DIR` variable to point to the folder where your dataset is stored.

3. **Run Training**:
   - After ensuring that the dataset is in the correct folder and the `DATA_DIR` is properly set, you can start training by running the following command:
     ```bash
     python src/train.py
     ```

4. **Output**:
   - The model will start training on the dataset and will output the best model checkpoint in the `results/checkpoints/` directory, where the results will be saved.

---

### Inference

To run inference on a new video, follow these steps:

1. **Set the Video Path**:
   - For inference, you need to specify the path of the video you want to process. Open the `main.py` file and modify the `VIDEO_PATH` variable to point to the desired video file.

2. **Run Inference**:
   - After setting the video path, run the following command to preprocess the video, detect faces, extract features, and generate predictions:
     ```bash
     python src/main.py
     ```

3. **Output**:
   - The script will preprocess the video, detect faces, extract features, and pass them through the Transformer model to output **emotion labels** and **probabilities** for the video.

---

## Future Plans

The current project focuses on **face emotion recognition** using video frames and Transformer-based models. However, there are several future plans to expand this into a comprehensive **multimodal emotion recognition system**.

### Planned Enhancements:
1. **Multimodal Fusion**: Integrating audio and text modalities along with video embeddings to enhance emotion recognition accuracy.
2. **Fine-Tuning**: Fine-tuning the models on **CMU-MOSEI** and **IEMOCAP** datasets for improved performance.
3. **Model Optimization**: Exploring optimization techniques for reducing model size and improving inference speed, making it more suitable for real-time applications.

---

## Contributing

Contributions are welcome! Please follow the standard [GitHub flow](https://guides.github.com/introduction/flow/) when contributing:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit them (`git commit -m "Add new feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Create a pull request.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- Face detection using [MTCNN](https://github.com/timesler/facenet-pytorch).
- Feature extraction [VGG19 model](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html).
- Classification using [Transformer model](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html).
