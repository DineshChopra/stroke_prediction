import sys
import os

import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:

  def __init__(self):
    pass

  def predict(self, features):
    try:
      model_path = os.path.join("artifacts","model.pkl")
      preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
      preprocessor = load_object(preprocessor_path)
      model = load_object(model_path)
      print('features -- ', features)
      data_scaled = preprocessor.transform(features)
      print('data scaled -- ', data_scaled)
      prediction = model.predict(data_scaled)
      return prediction
    except Exception as e:
      raise CustomException(e, sys)

class CustomData:
  def __init__(self, 
  gender: str,
  age: float,
  hypertension: int,
  heart_disease: int,
  ever_married: str, # Need to fill in html
  work_type: str, # Need to fill in html
  Residence_type: str, # Need to fill in html
  avg_glucose_level: float,
  bmi: float,
  smoking_status: str):
    '''
      Male,
      67.0,
      0,
      1,
      Yes,Private,Urban,
      228.69,36.6,formerly smoked,1
    '''
    self.gender = gender
    self.age = age
    self.hypertension = hypertension
    self.heart_disease = heart_disease
    self.ever_married = ever_married
    self.work_type = work_type
    self.Residence_type = Residence_type
    self.avg_glucose_level = avg_glucose_level
    self.bmi = bmi
    self.smoking_status = smoking_status
    
  def get_data_as_data_frame(self):
    try:
      custom_data_input_dict = {
        "gender" : [self.gender],
        "age" : [self.age],
        "hypertension" : [self.hypertension],
        "heart_disease" : [self.heart_disease],
        "ever_married" : [self.ever_married],
        "work_type" : [self.work_type],
        "Residence_type" : [self.Residence_type],
        "avg_glucose_level" : [self.avg_glucose_level],
        "bmi" : [self.bmi],
        "smoking_status" : [self.smoking_status],
      }
      print('custom_data_input_dict -- ', custom_data_input_dict)
      return pd.DataFrame(custom_data_input_dict)
    except Exception as e:
      raise CustomException(e, sys)