import os
import sys

from src.logger import logging
from src.exception import CustomException

from shutil import copyfile
from sklearn.model_selection import train_test_split

from src.components.data_validation import DataValidation
from src.components.model_trainer import ModelTrainer

class DataIngestionConfig:
    train_data_path = os.makedirs(os.path.join('artifacts', 'train'))
    test_data_path = os.makedirs(os.path.join('artifacts', 'test'))

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, root_path, dataset_folder, split_size):
        try:
            # os.makedirs(os.path.join(root_path, 'train'))
            # os.makedirs(os.path.join(root_path, 'test'))

            # Get the list of subdirectories (each subdirectory is a class)
            class_folders = os.listdir(dataset_folder)

            for class_folder in class_folders:
                os.makedirs(os.path.join(root_path, 'train', class_folder))
                os.makedirs(os.path.join(root_path, 'test', class_folder))

                class_path = os.path.join(dataset_folder, class_folder)

                images = os.listdir(class_path)

                train_images, test_images = train_test_split(images, test_size=split_size, random_state=42)

                for img in train_images:
                    source = os.path.join(class_path, img)
                    destination = os.path.join(root_path, 'train', class_folder, img)
                    copyfile(source, destination)

                for img in test_images:
                    source = os.path.join(class_path, img)
                    destination = os.path.join(root_path, 'test', class_folder, img)
                    copyfile(source, destination)

            return train_images, test_images

        except Exception as e:
            raise(CustomException(e, sys))
            

if __name__ == '__main__':

    split_size = 0.1
    root_dir = 'artifacts'
    dataset_folder = 'notebook\Fruits dataset'
    # initiate_data_ingestion(root_dir, dataset_folder, split_size)
    
    module = DataIngestion()
    module.initiate_data_ingestion(root_dir, dataset_folder, split_size)

    train_data = 'artifacts/test/'
    test_data = 'artifacts/train/'
    
    data_validation = DataValidation()
    train_dataset, valid_dataset = data_validation.initiate_data_validation(train_data, test_data)

    model = ModelTrainer()
    print(model.evalute_model(train_dataset, valid_dataset))