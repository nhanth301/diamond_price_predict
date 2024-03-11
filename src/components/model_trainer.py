import pandas as pd
import numpy as np
from src.logger.logging import logging
from src.exception.exception import CustomException
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.utils import save_object

from sklearn.linear_model import LinearRegression, Ridge,Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_model(self, X_train, y_train, X_test, y_test, models):
        try:
            report = {}
            for model_name in models.keys():
                model = models[model_name]
                model.fit(X_train,y_train)
                y_test_pred = model.predict(X_test)
                test_r2_score = r2_score(y_test,y_test_pred)
                report[model_name] = test_r2_score
            return report
        
        except Exception as e:
            logging.info("Exception occured during model training")
            raise CustomException(e,sys)

    
    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
            'LinearRegression' : LinearRegression(),
            'Lasso' : Lasso(),
            'Ridge' : Ridge(),
            'Elasticnet' : ElasticNet(),
            'RandomForest': RandomForestRegressor(),
            'XGBoost' : XGBRegressor()
        }
            
            model_report:dict= self.evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            # To get best model score from dictionary 
            best_model_score = max(model_report.values())

            best_model_name = [key for key in model_report.keys() if model_report[key] == best_model_score][0]
            
            best_model = models[best_model_name]

            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )
          

        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)

        
    