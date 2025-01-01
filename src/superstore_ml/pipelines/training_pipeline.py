
from src.superstore_ml.components.data_ingestion import DataIngestion
from src.superstore_ml.components.data_transformation import DataTransformation
import os 
import sys
from src.superstore_ml.logger import logging
from src.superstore_ml.exception import customexception
from src.superstore_ml.components.model_trainer import ModelTrainer
import pandas as pd
obj=DataIngestion()
obj.initiate_data_ingestion()


def start_data_transformation(self,train_data_path,test_data_path):     
    try:
        transformation = DataTransformation()
        train_arr,test_arr=transformation.initiate_data_transformation(train_data_path,test_data_path)
        return train_arr,test_arr
    except Exception as e:
        raise customexception(e,sys)
            
            
            
        
    
def start_model_training(self,train_arr,test_arr):
    try:
        model_trainer=ModelTrainer()
        model=model_trainer.initiate_model_training(train_arr,test_arr)
        return model
    except Exception as e:
        raise customexception(e,sys)
