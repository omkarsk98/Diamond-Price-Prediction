""" 
  Data Ingestion:
    Input: Raw data from any source. Cloud Storage, Database, local file system, etc.
    Output: Data in a format that can be used by the Data Processing component.
    Steps:
      1. Split into train and test sets

"""
import os
import sys
from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

# Initialize the data ingestion configuration

@dataclass
class DataIngestionConfig:
    """Data ingestion configuration"""
    trainDataPath: str = os.path.join("artifacts", "train.csv")
    testDataPath: str = os.path.join("artifacts", "test.csv")
    rawDataPath: str = os.path.join("artifacts", "raw.csv")

# Initialize the data ingestion class
class DataIngestion:
    """Data ingestion class"""
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiateDataIngestion(self):
      logging.info("Data ingestion has started")
      try:
        df = pd.read_csv(os.path.join("notebooks/data", "gemstone.csv"))
        logging.info("Data read in pandas dataframe")

        os.makedirs(os.path.dirname(self.config.rawDataPath), exist_ok=True)
        df.to_csv(self.config.rawDataPath, index = False)

        logging.info("Raw data is created")

        train, test = train_test_split(df, test_size=0.3, random_state=42)

        train.to_csv(self.config.trainDataPath, index = False, header = True)
        test.to_csv(self.config.testDataPath, index = False, header = True)

        logging.info("Train and test data is created")

        return (self.config.trainDataPath, self.config.testDataPath)

      except Exception as e:
        logging.info('Data ingestion failed')
        raise CustomException(e,sys)