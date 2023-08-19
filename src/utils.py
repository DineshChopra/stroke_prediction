import os
import sys

import numpy as np
import pandas as pd
import dill
from src.logger import logging

from src.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
      dill.dump(obj, file_obj)
  except Exception as e:
    raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, param):
  try:
    report = {}
    model_names = list(models.keys())
    for model_name in model_names:
      model = models[model_name]
      # Hyper parameter tuning
      hyp_param = param[model_name]
      gs = GridSearchCV(model, param_grid=hyp_param, cv=5)
      gs.fit(X_train, y_train)
      # Find out best parameters
      best_params = gs.best_params_
      logging.info('best parameters: ', best_params)

      model.set_params(**best_params) 
      model.fit(X_train, y_train) # Train Model

      y_train_pred = model.predict(X_train)
      y_test_pred = model.predict(X_test)

      train_model_score = accuracy_score(y_train, y_train_pred)
      test_model_score = accuracy_score(y_test, y_test_pred)
      report[model_name] = test_model_score

    return report

  except Exception as e:
    raise CustomException(e, sys)
