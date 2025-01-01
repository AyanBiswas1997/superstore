import pandas as pd
import numpy as np
import os 
import sys
from dataclasses import dataclass
from src.superstore_ml.logger import logging
from src.superstore_ml.exception import customexception

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,OrdinalEncoder
from src.superstore_ml.utils.utils import save_object




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()


    def get_data_transformation(self):
        try:
            logging.info("Data transformation started")
            categorical_column=['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type',
       'Outlet_Type']
            numerical_column=['Item_Weight', 'Item_Visibility', 'Item_MRP',
       'Outlet_Establishment_Year']
            Item_Fat_Content_category=['Low Fat', 'Regular', 'low fat', 'LF', 'reg']
            Item_Type_category=['Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
       'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast',
       'Health and Hygiene', 'Hard Drinks', 'Canned', 'Breads',
       'Starchy Foods', 'Others', 'Seafood']
            Outlet_Size_category=['Medium', 'High', 'Small']
            Outlet_Location_Type_category=['Tier 1', 'Tier 3', 'Tier 2']
            Outlet_Type_category=['Supermarket Type1', 'Supermarket Type2', 'Grocery Store',
       'Supermarket Type3']
            
        #pipeline started
            logging.info("pipeline started")
        #numeric pipeline
            num_pipeline=Pipeline(
                steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("scaler",StandardScaler())
            ])
        #categorical pipeline
            cat_pipeline=Pipeline(
                steps=[
            ("imputer",SimpleImputer(strategy="most_frequent")),
            ("ordinal",OrdinalEncoder(categories=[Item_Fat_Content_category,Item_Type_category,Outlet_Size_category,Outlet_Location_Type_category,Outlet_Type_category]))

            ]) 

        #preprocessor 
            preprocessor=ColumnTransformer(
                steps=[
            ("num",num_pipeline,numerical_column),
            ("cat",cat_pipeline,categorical_column)
            ])
            
            return preprocessor


            

            



        except Exception as e:
            logging.error(f"Error occured in data transformation {e}")
            raise customexception(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data")
            logging.info(f"Train dataframe head: \n{train_df.head().to_string()}")
            logging.info(f"Test dataframe head: \n{test_df.head().to_string()}")

            preprocessor_obj=self.get_data_transformation()

            target_column="Item_Outlet_Sales"
            drop_columns=["Item_Identifier","target_column","Outlet_Identifier"]



            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column] 

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column]

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj()

            )
            logging.info("Preprocessor object saved in artifacts folder")
            return (
                train_arr,
                test_arr
            )



        except Exception as e:
            logging.error(f"Error occured in data transformation {e}")
            raise customexception(e,sys)







