import os, sys
import numpy as np
import pandas as pd

import dill
import tensorflow as tf

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        obj.save(file_path)
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        model = tf.keras.models.load_model(file_path)
        return model

    except Exception as e:
        raise CustomException(e, sys)