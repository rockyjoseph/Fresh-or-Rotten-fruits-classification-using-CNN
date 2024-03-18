import sys
import datetime
import pandas as pd

import cv2
from PIL import Image, ImageOps

from src.utils import load_object
from src.exception import CustomException

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, image_data):
        try:
            size = (256,256)
            model_path = 'artifacts/models/fruits/'
            model = load_object(file_path = model_path)
            
            image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
            image = np.asarray(image)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            img_reshape = img[np.newaxis, ...]

            pred = model.predict(img_reshape)

            return pred

        except Exception as e:
            raise CustomException(e, sys)