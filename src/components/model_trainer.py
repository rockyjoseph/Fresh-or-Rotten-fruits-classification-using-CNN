import os, sys
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from src.utils import save_object

import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras import models, layers

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts','models','fruits')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, train_data, val_data):
        try:
            input_shape = (256, 256, 3)
            BATCH_SIZE = 64
            n_classes = 6
            EPOCHS = 10

            logging.info('Starting Model building')

            model = models.Sequential([
                layers.Conv2D(16, (5,5), padding='valid', activation='relu', kernel_regularizer=l2(0.00005), input_shape=input_shape),
                layers.MaxPooling2D((2,2)),
                layers.BatchNormalization(),
        
                layers.Conv2D(32, (5,5), padding='valid', activation='relu', kernel_regularizer=l2(0.00005), input_shape=input_shape),
                layers.MaxPooling2D((2,2)),
                layers.BatchNormalization(),
        
                layers.Conv2D(64, (3,3), padding='valid', activation='relu', kernel_regularizer=l2(0.00005), input_shape=input_shape),
                layers.MaxPooling2D((2,2)),
                layers.BatchNormalization(),
                
                layers.Conv2D(128, (3,3), padding='valid', activation='relu', kernel_regularizer=l2(0.00005), input_shape=input_shape),
                layers.MaxPooling2D((2,2)),
                layers.BatchNormalization(),
                
                layers.Conv2D(256, (3,3), padding='valid', activation='relu', kernel_regularizer=l2(0.00005), input_shape=input_shape),
                layers.MaxPooling2D((2,2)),
                layers.BatchNormalization(),
                
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dense(n_classes, activation='softmax')
            ])

            model.build(input_shape=(None, 256, 256, 3))
            
            model.summary()

            model.compile(
                optimizer = 'adam',
                loss = tf.keras.losses.CategoricalCrossentropy(),
                metrics=['accuracy']
            )

            history = model.fit(
                train_data,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                verbose=1,
                validation_data=val_data
            )

            logging.info('Finished model building!')

            score = model.evaluate(val_data)

            save_object(
                file_path= self.model_trainer_config.trained_model_path,
                obj = model
            )

            logging.info(f"Best model for prediction")

            return score

        except Exception as e:
            raise CustomException(e, sys)