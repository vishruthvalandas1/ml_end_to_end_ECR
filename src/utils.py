import os
import sys
from tabnanny import verbose
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import dill 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models,params):
    try:
        report = {}
        for i in range(len(list(models))):
            
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]
            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            # Train model
            # model.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = (test_model_score,gs.best_params_)
            logging.info(f"{list(models.keys())[i]} : {train_model_score}, {test_model_score}, {gs.best_params_}")
        
        return report
    except Exception as e:
        logging.info("Exception occurred during model training")
        raise CustomException(e, sys)