import os
import sys

import numpy as np
import pandas as pd
import pickle
from src.logger import logging

from src.exception import CustomException
from sklearn.utils import resample

def save_object(file_path, obj):
  try:
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    with open(file_path, "wb") as file_obj:
      pickle.dump(obj, file_obj)
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