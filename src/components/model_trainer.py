import os 
import sys

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    model_trainer_file_path=os.path.join('arifacts','model_trainer.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("split the training and test input data")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                'Linear Regression':LinearRegression(),
                'AdaBoost Regressor':AdaBoostRegressor(),
                'K-Nearest Neighbours':KNeighborsRegressor(),
                'Decision Tree Regressor':DecisionTreeRegressor(),
                'XG Boost Regressor':XGBRegressor(),
                'Random Forest Regressor':RandomForestRegressor(),
                'Gradient Boosting':GradientBoostingRegressor(),
                'Cat Boost Regressor':CatBoostRegressor(verbose=False)
            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No Best Model')
            logging.info('Best Model found')

            save_object(
                file_path=self.model_trainer_config.model_trainer_file_path,
                obj=best_model
            )

            predicted_score=best_model.predict(X_test)
            r2_scores=r2_score(y_test,predicted_score)

            return r2_scores
            
            
        except Exception as e:
            raise CustomException(e,sys)
            


