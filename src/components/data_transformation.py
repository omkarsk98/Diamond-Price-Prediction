from sklearn.impute import SimpleImputer ## Handling Missing Values
from sklearn.preprocessing import StandardScaler # Handling Feature Scaling
from sklearn.preprocessing import OrdinalEncoder # Ordinal Encoding
## pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd

from src.exception import CustomException
from src.logger import logging
import sys, os
from dataclasses import dataclass
from src.utils import saveObject

@dataclass
class DataTransformationConfig:
    """Data transformation configuration"""
    preprocessorFilePath = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
  def __init__(self):
    self.config = DataTransformationConfig()
  
  def getDataTransformer(self):
    try:
      logging.info("Data transformation initiated")
      
      # split numerical and categorical features
      categorical_cols = ['cut', 'color', 'clarity']
      numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
      
      # Define the custom ranking for each ordinal variable
      cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
      color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
      clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

      logging.info("Data Transformation pipeline initiated")

      ## Numerical Pipeline
      num_pipeline = Pipeline(
          steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
      )

      # Categorigal Pipeline
      cat_pipeline = Pipeline(
          steps=[
              ("imputer", SimpleImputer(strategy="most_frequent")),
              (
                  "ordinalencoder",
                  OrdinalEncoder(
                      categories=[cut_categories, color_categories, clarity_categories]
                  ),
              ),
              ("scaler", StandardScaler()),
          ]
      )

      preprocessor = ColumnTransformer(
          [
              ("num_pipeline", num_pipeline, numerical_cols),
              ("cat_pipeline", cat_pipeline, categorical_cols),
          ]
      )
      logging.info("Data Transformation pipeline completed")
      return preprocessor

    except Exception as e:
      logging.info('Exception occured in data transformation')
      raise CustomException(e,sys)

  def initiateDataTransformer(self, trainDataPath, testDataPath):
    try:
      trainDF = pd.read_csv(trainDataPath)
      testDF = pd.read_csv(testDataPath)
      
      preProcessor = self.getDataTransformer()
      targetCol = "price"
      dropColumns = ["id", targetCol]

      # Split into input and target for train data
      inputFeatureTrainDF = trainDF.drop(dropColumns, axis=1)
      targetFeatureDF = trainDF[targetCol]

      # split into input and target for test data
      inputFeatureTestDF = testDF.drop(dropColumns, axis=1)
      targetFeatureTestDF = testDF[targetCol]

      # Transform the train data
      inputFeatureTrainArr = preProcessor.fit_transform(inputFeatureTrainDF)
      inputFeatureTestArr = preProcessor.transform(inputFeatureTestDF)

      trainArr = np.c_[inputFeatureTrainArr, np.array(targetFeatureDF)]
      testArr = np.c_[inputFeatureTestArr, np.array(targetFeatureTestDF)]

      saveObject(self.config.preprocessorFilePath, preProcessor)

      return (trainArr, testArr, self.config.preprocessorFilePath)

    except Exception as e:
      logging.info('Exception occured in data transformation')
      raise CustomException(e,sys)
