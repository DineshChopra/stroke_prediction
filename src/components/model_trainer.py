import os
import sys
from dataclasses import dataclass

from sklearn.model_selection import train_test_split, GridSearchCV
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

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
  trained_model_filepath = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def get_best_hyperparameters(self, model_info, X_train, y_train):
    model = model_info['model']
    param_grid = model_info['params']

    gs = GridSearchCV(model, param_grid, cv=5)
    gs.fit(X_train, y_train)
    # Find out best parameters
    return gs.best_params_

  def retrain_model_on_best_hyperparameters(self, model, best_params, X_train, y_train):
    model.set_params(**best_params) 
    model.fit(X_train, y_train) # Train Model
    return model

  def get_accuracy_score(self, model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    return accuracy_score(y_test, y_test_pred)

  def evaluate_model(self, X_train, y_train, X_test, y_test, models, param):
    try:
      report = {}
      model_names = list(models.keys())
      for model_name in model_names:
        model = models[model_name]

        # Hyper parameter tuning
        hyp_param = param[model_name]

        # Find out best parameters
        best_params = self.get_best_hyperparameters(model, hyp_param, X_train, y_train)
        logging.info('best parameters: ', best_params)

        test_model_score = self.get_accuracy_score(model, best_params, X_train, y_train, X_test, y_test)
        report[model_name] = test_model_score

      return report

    except Exception as e:
      raise CustomException(e, sys)

  def get_best_model_name(self, model_acc_dict):
    best_model_score = max(sorted(model_acc_dict.values()))
    best_model_name = list(model_acc_dict.keys())[
      list(model_acc_dict.values()).index(best_model_score)
    ]
    return best_model_name

  def save_best_model(self, best_model):
    save_object(
      file_path=self.model_trainer_config.trained_model_filepath,
      obj=best_model
    )

