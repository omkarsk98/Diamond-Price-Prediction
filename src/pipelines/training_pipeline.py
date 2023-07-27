import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    ingest = DataIngestion()
    trainDataPath, testDataPath = ingest.initiateDataIngestion()
    print(trainDataPath, testDataPath)