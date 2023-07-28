import numpy as np
import pandas as pd
import os
import sys

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import saveObject, evaluateModel
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    """Model trainer configuration"""
    trainedModelPath: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
  def __init__(self):
    self.config = ModelTrainerConfig()

  def initiateModelTraining(self, trainArr, testArr):
    try:
      logging.info("Model training initiated")
      X_train, y_train = trainArr[:, :-1], trainArr[:, -1]
      X_test, y_test = testArr[:, :-1], testArr[:, -1]

      models = {
          "LinearRegression": LinearRegression(),
          "Ridge": Ridge(),
          "Lasso": Lasso(),
          "ElasticNet": ElasticNet(),
          "RandomForestRegressor": RandomForestRegressor(),
          # "KNeighborsRegressor": KNeighborsRegressor(),
          "DecisionTreeRegressor": DecisionTreeRegressor(),
      }

      report = evaluateModel(X_train, y_train, X_test, y_test, models)
      print("============================================================")
      logging.info(f"Model evaluation report: {report}")

      # get the best model score from report
      bestModelScore = np.array(list(report.values()))[:, 2]
      bestModelName = list(report.keys())[np.argmax(bestModelScore)]
      logging.info(f"Best model name: {bestModelName}")

      saveObject(self.config.trainedModelPath, models[bestModelName])

    except Exception as e:
      logging.info("Model training failed")
      raise CustomException(e, sys)