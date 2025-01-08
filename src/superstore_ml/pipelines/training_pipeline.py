
from src.superstore_ml.components.data_ingestion import DataIngestion
from src.superstore_ml.components.data_transformation import DataTransformation
import os 
import sys
from src.superstore_ml.logger import logging
from src.superstore_ml.exception import customexception
from src.superstore_ml.components.model_trainer import ModelTrainer
import pandas as pd
obj=DataIngestion()
train_data_path,test_data_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
train_arr,test_arr=data_transformation.initialize_data_transformation(train_data_path,test_data_path)



model_trainer = ModelTrainer()
model_trainer.initiate_model_training(train_arr, test_arr)