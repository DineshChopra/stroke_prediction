import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
  trained_model_filepath = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_arr, test_arr):
    try:
      logging.info("Split training train and test input data")
      X_train, y_train, X_test, y_test = (
        train_arr[:, :-1],
        train_arr[:, -1],
        test_arr[:, :-1],
        test_arr[:, -1]
      )

      models = {
        "Decision Tree": DecisionTreeClassifier(criterion='entropy'),
        "Random Forest": RandomForestClassifier(n_estimators=150, criterion='entropy', random_state=123)
      }

      model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, 
                                          X_test=X_test, y_test=y_test,
                                          models=models)

      # Find out Best model
      best_model_score = max(sorted(model_report.values()))
      best_model_name = list(model_report.keys())[
        list(model_report.values()).index(best_model_score)
      ]
      best_model = models[best_model_name]
      logging.info(f"Best Model: {best_model} and Model accuracy : {best_model_score}")
      if best_model_score < 0.8:
        raise CustomException("No good model found")
      
      logging.info("Best model on train and test dataset")

      save_object(
        file_path=self.model_trainer_config.trained_model_filepath,
        obj=best_model
      )

      y_test_pred = best_model.predict(X_test)
      accuracy = accuracy_score(y_test, y_test_pred)
      return accuracy
    except Exception as e:
      raise CustomException(e, sys)
