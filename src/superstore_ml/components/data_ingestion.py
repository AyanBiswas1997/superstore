from src.superstore_ml.logger import logging
from src.superstore_ml.exception import customexception
import pandas as pd
import os

import numpy
import sys
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from pathlib import Path


class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifatcts","raw.csv")
    train_data_path:str=os.path.join("artifatcts","train.csv")
    test_data_path:str=os.path.join("artifatcts","test.csv")





class DataIngestion:
    def __init__(self):
        self.ingesion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")

        try:
            data=pd.read_csv(os.path.join("notebooks/data","Train.csv"))
            logging.info("Data ingestion completed")

            os.makedirs(os.path.dirname(os.path.join(self.ingesion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingesion_config.raw_data_path,index=False)
            logging.info("Data saved in raw data path")

            logging.info("train test split started")
            
            train_data,test_data=train_test_split(data,test_size=0.2,random_state=42)
            logging.info("train test split completed")
            train_data.to_csv(self.ingesion_config.train_data_path,index=False)
            test_data.to_csv(self.ingesion_config.test_data_path,index=False)
            logging.info("Data saved in train and test data path")
            return(
                self.ingesion_config.train_data_path,
                self.ingesion_config.test_data_path
            )


        except Exception as e:
            logging.error(f"Error occured in data ingestion {e}")
            raise customexception(e,sys)
