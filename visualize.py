from src.utils.models import FaceEmotionModel

from torchinfo import summary


model = FaceEmotionModel()
summary(model, input_size=(32, 3, 224, 224))
