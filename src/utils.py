import os
import sys
import pickle
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def saveObject(filePath, obj):
  
  try:
    dirPath = os.path.dirname(filePath)
    os.makedirs(dirPath, exist_ok=True)
    with open(filePath, "wb") as f:
      pickle.dump(obj, f)

  except Exception as e:
    logging.info("Object saving failed")
    raise CustomException(e, sys)

def evaluateModel(xTrain, yTrain, xTest, yTest, models):
  try:
    logging.info("Model evaluation initiated")

    report = {}
    i = 0
    for name, model in models.items():
      model.fit(xTrain, yTrain)
      
      yPred = model.predict(xTest)
      
      mse = mean_squared_error(yTest, yPred)
      mae = mean_absolute_error(yTest, yPred)
      r2 = r2_score(yTest, yPred)
      
      report[list(models.keys())[i]] = [mse, mae, r2]
      i += 1
      logging.info(f"Evaluated {name}")

    logging.info("Model evaluation completed")
    return report

  except Exception as e:
    logging.info("Model evaluation failed")
    raise CustomException(e, sys)

def loadObject(filePath):
  try:
    with open(filePath, "rb") as f:
      obj = pickle.load(f)
    return obj

  except Exception as e:
    logging.info("Object loading failed")
    raise CustomException(e, sys)