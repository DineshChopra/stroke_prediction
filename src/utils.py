import os
import sys

import numpy as np
import pandas as pd
import pickle
from src.logger import logging

from src.exception import CustomException
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
      pickle.dump(obj, file_obj)
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

      # y_train_pred = model.predict(X_train)
      y_test_pred = model.predict(X_test)

      # train_model_score = accuracy_score(y_train, y_train_pred)
      test_model_score = accuracy_score(y_test, y_test_pred)
      report[model_name] = test_model_score

    return report

  except Exception as e:
    raise CustomException(e, sys)

def load_object(file_path):
  try:
    # with open(file_path, 'rb') as file_obj:
    with open(file_path, "rb") as file_obj:
      return pickle.load(file_obj)
  except Exception as e:
    raise CustomException(e, sys)

def resample_unbalance_data(df):
  df_0 = df[df.iloc[:, -1] == 0]
  df_1 = df[df.iloc[:, -1] == 1]
  #  It creates duplicate records in random fashion
  df_1 = resample(df_1, replace=True, n_samples=df_0.shape[0], random_state=42)
  df = np.concatenate((df_0, df_1))
  df = pd.DataFrame(df)
  df.columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married','work_type', 
              'Residence_type', 'avg_glucose_level', 'bmi','smoking_status', 'stroke']
  return df

def data_cleaning(df):
  # Remove Unnecessary Column
  df.drop(columns=['id'], axis=1, inplace=True)

  # Remove dupliates
  df.drop_duplicates(inplace=True)

  # Remove Null records
  df.dropna(inplace=True)
  return df