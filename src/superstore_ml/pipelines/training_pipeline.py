
from src.superstore_ml.components.data_ingestion import DataIngestion
import os 
import sys
from src.superstore_ml.logger import logging
from src.superstore_ml.exception import customexception
import pandas as pd
obj=DataIngestion()
obj.initiate_data_ingestion()