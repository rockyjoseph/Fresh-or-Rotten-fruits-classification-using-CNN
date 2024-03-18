import os, sys
import numpy as np
import pandas as pd

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

@dataclass
class DataValidation:
    def __init__(self):
        pass

    def initiate_data_validation(self, train_data, test_data):
        '''
            Applying Data Augmentation on Train, Test and Validation data
        '''

        try:
            IMAGE_SIZE = 256
            BATCH_SIZE = 64
            CHANNELS = 6

            train_datagen = ImageDataGenerator()

            train_generator = train_datagen.flow_from_directory(
                train_data,
                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                batch_size=BATCH_SIZE,
                shuffle=True
            )

            test_datagen = ImageDataGenerator()

            test_generator = test_datagen.flow_from_directory(
                test_data,
                shuffle=True,
                batch_size=64,
                target_size=(IMAGE_SIZE, IMAGE_SIZE)
            )

            return train_generator, test_generator

        except Exception as e:
            CustomException(e, sys)