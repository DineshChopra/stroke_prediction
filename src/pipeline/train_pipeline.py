import sys
import os

import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient

timestamp = datetime.now().timestamp()
      
tag_name = f"stroke_predictor_{timestamp}"
class TrainPipeline:
  
  def __init__(self):
    pass

  def train(self):

      data_ingestion = DataIngestion()
      train_data, test_data = data_ingestion.initiate_data_ingestion()
      
      data_transformation = DataTransformation()
      train_arr, test_arr, preprocessor_file_path = data_transformation.initiate_data_transformation(train_data, test_data)

      X_train, y_train, X_test, y_test = (
        train_arr[:, :-1],
        train_arr[:, -1],
        test_arr[:, :-1],
        test_arr[:, -1]
      )
      model_infos = [
        {
          "name": "Decision Tree",
          "model": DecisionTreeClassifier(),
          "params": {
            'criterion':['entropy'],
          }
        },
        {
          "name": "Random Forest",
          "model": RandomForestClassifier(random_state=123),
          "params": {
            'criterion':['entropy'],
            'n_estimators': [10, 20, 30]
          }
        }
      ]

      model_trainer = ModelTrainer()
      model_report = {}
      best_models = {}
      
      for model_info in model_infos:
        print('model_iinfo -- ', model_info)
        with mlflow.start_run(nested=True):
          mlflow.log_artifact(local_path=preprocessor_file_path, artifact_path="preprocessor")
          mlflow.set_tag("developer", "Dinesh Chopra")
          mlflow.set_tag("tag_name", tag_name)
          mlflow.log_param("Train Dataset Path", train_data)
          mlflow.log_param("Test Dataset Path", test_data)

          best_params = model_trainer.get_best_hyperparameters(model_info, X_train, y_train)
          # Retrain model on best_hyperparameters
          model = model_info['model']
          model_name = model_info['name']
          
          mlflow.log_param("Model Name", model_name)
          mlflow.log_params(best_params)

          best_model = model_trainer.retrain_model_on_best_hyperparameters(model, best_params, X_train, y_train)
          mlflow.sklearn.log_model(best_model, artifact_path="model")
          best_models[model_name] = best_model

          accuracy = model_trainer.get_accuracy_score(best_model, X_test, y_test)
          model_report[model_name] = accuracy
          mlflow.log_metric("accuracy", accuracy)
          print(f"Model: {model_name}, accuracy: {accuracy}")


  def register_best_model(self):
    client = MlflowClient()
    best_performace_runs = client.search_runs(
        experiment_ids="1",
        filter_string=f"tag.tag_name = '{tag_name}' ",
        order_by=["metrics.accuracy DESC"],
    )[0]
    model_uri = best_performace_runs.info.run_id
    print('model_uri --- ', model_uri)
    mlflow.register_model(
      model_uri=f"runs:/{model_uri}/model", name="stroke_predictor_classifier"
    )
if __name__ == '__main__':
  
  mlflow.set_tracking_uri("sqlite:///mlflow.db")
  mlflow.set_experiment("stroke-prediction-experiment")
  try:
    train_pipeline = TrainPipeline()
    train_pipeline.train()
    train_pipeline.register_best_model()

  except Exception as e:
    raise CustomException(e, sys)
  