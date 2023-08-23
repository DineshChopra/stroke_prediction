import pickle
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

from src.utils import load_object
from src.pipeline.predict_pipeline import CustomData

app = FastAPI(title="Stroke Prediction App")

# Represents a particular stroke (or datapoint)
class Stroke(BaseModel):
  gender: str
  age: float
  hypertension: int
  heart_disease: int
  ever_married: str # Need to fill in html
  work_type: str # Need to fill in html
  Residence_type: str # Need to fill in html
  avg_glucose_level: float
  bmi: float
  smoking_status: str

  def get_stroke_as_data_frame(self):
    try:
      data = {
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
      
      print('data -- ', data)
      return pd.DataFrame(data)
    except Exception as e:
      raise CustomException(e, sys)


preprocessor_path = 'artifacts/preprocessor.pkl'
model_path = 'artifacts/model.pkl'

@app.on_event("startup")
def load_preprocessor_and_classifier():
    # Load classifier from pickle file
    global preprocessor
    global model

    preprocessor = load_object(preprocessor_path)
    model = load_object(model_path)


@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:80/docs"


@app.post("/predict")
def predict(stroke: Stroke):
  print("stroke -- ", stroke)
  
  pred_df = stroke.get_stroke_as_data_frame()
  print('pred_df ---->>>>>> --- ', pred_df)

  data_scaled = preprocessor.transform(pred_df)
  print('data scaled -- ', data_scaled)
  predictions = model.predict(data_scaled)
  return {"Prediction": predictions[0]}

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080, debug=True)
