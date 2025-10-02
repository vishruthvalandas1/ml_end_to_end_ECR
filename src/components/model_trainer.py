from ast import Param
import os
from sqlite3 import paramstyle
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model
from catboost import CatBoostRegressor
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False)
            }

            Param={
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Decision Tree": {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]},
                "Gradient Boost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "KNN": {"n_neighbors": [5, 11, 15, 20]},
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                }

            }
            logging.info("Model Training started")
            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,params=Param)
            logging.info(f"Model Report : {[model[0] for model in model_report.values()]}")
            best_model_score = max(sorted([model[0] for model in model_report.values()]))
            best_model_name = list(model_report.keys())[
                list([model[0] for model in model_report.values()]).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException("No best model found",error_detail=None)
            logging.info(f"Best found model on both training and testing dataset")
            save_object(
                    file_path=self.model_trainer_config.trained_model_file_path,
                    obj=best_model
                )
            logging.info(f"Best model found on training dataset")
            predicted= best_model.predict(X_test)
            r2_score_square= r2_score(y_test, predicted)
            return r2_score_square,best_model

            
        except Exception as e:
            raise CustomException(e, sys)
        


