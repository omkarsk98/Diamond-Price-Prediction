import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    ingest = DataIngestion()
    trainDataPath, testDataPath = ingest.initiateDataIngestion()
    print(trainDataPath, testDataPath)
    dataTransformer = DataTransformation()
    trainArr, testArr, preprocessorPath = dataTransformer.initiateDataTransformer(trainDataPath, testDataPath)

    modelTrainer = ModelTrainer()
    modelTrainer.initiateModelTraining(trainArr, testArr)