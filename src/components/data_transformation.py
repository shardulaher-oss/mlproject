# Data evaluate
import os
from dataclasses import dataclass

import sys
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file=os.path.join('arifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numeric_features=['reading_score','writing_score']
            cat_features=[
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            numeric_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scalar',StandardScaler())
                ]
            ) 
            category_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder(handle_unknown='ignore')),
                    ('scalar',StandardScaler(with_mean=False))  
                ]
            )
            preprocessor=ColumnTransformer(    #combines both features
                [
                    ('numeric_pipeline',numeric_pipeline,numeric_features),
                    ('category_pipeline',category_pipeline,cat_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_data,test_data):

        try:
            train_df=pd.read_csv(train_data)
            test_df=pd.read_csv(test_data)
            logging.info('Read the train and test data completed')
            logging.info('Obtaining preprocessing obj')
            
            preprocessing_obj=self.get_data_transformer_obj()
            target_feature='math_score'
            numeric_feature=['reading_score','writing_score']

            input_feature_train_df=train_df.drop(columns=[target_feature],axis=1)
            target_feature_train_df=train_df[target_feature]
            input_feature_test_df=test_df.drop(columns=[target_feature],axis=1)
            target_feature_test_df=test_df[target_feature]
            logging.info('Applying preprocessing obj on training and testing dataframe')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info('Saved preprocessing obj')
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file,
                obj=preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file
            )
           
        except Exception as e:
            raise CustomException(e,sys)
            
