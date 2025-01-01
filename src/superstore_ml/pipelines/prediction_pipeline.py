
import os
import sys
import pandas as pd
from src.superstore_ml.exception import customexception
from src.superstore_ml.logger import logging
from src.superstore_ml.utils.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise customexception(e,sys)
    
  
    
class CustomData:
    def __init__(self,
                 Item_Weight:float,
                 Item_Fat_Content:str,
                 Item_Visibility:float,                     		
                 Item_MRP:float,
                 Outlet_Establishment_Year:float,                                                           
                 Item_Type:str,
                 Outlet_Size:str,
                 Outlet_Location_Type:str,
                 Outlet_Type:str):
        
        self.Item_weight=Item_Weight
        self.Item_fat_Content=Item_Fat_Content
        self.Item_visibility=Item_Visibility
        self.Item_MRP=Item_MRP
        self.Outlet_Establishment_Year=Outlet_Establishment_Year
        self.Item_Type=Item_Type
        self.Outlet_Size=Outlet_Size
        self.Outlet_Location_Type=Outlet_Location_Type
        self.Outlet_Type=Outlet_Type
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'Item_weight':[self.Item_weight],
                    'Item_fat_Content':[self.Item_fat_Content],
                    'Item_visibility':[self.Item_visibility],
                    'Item_MRP':[self.Item_MRP],
                    'Outlet_Establishment_Year':[self.Outlet_Establishment_Year],
                    'Item_Type':[self.Item_Type],
                    'Outlet_Size':[self.Outlet_Size],
                    'Outlet_Location_Type':[self.Outlet_Location_Type],
                    'Outlet_Type':[self.Outlet_Type]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise customexception(e,sys)