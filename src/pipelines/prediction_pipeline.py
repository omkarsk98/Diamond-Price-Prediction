import sys
import os
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import loadObject


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            preprocessorPath = os.path.join("artifacts", "preprocessor.pkl")
            modelPath = os.path.join("artifacts", "model.pkl")

            preprocessor = loadObject(preprocessorPath)
            model = loadObject(modelPath)

            scaledData = preprocessor.transform(features)

            pred = model.predict(scaledData)
            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        carat: float,
        depth: float,
        table: float,
        x: float,
        y: float,
        z: float,
        cut: str,
        color: str,
        clarity: str,
    ):
        self.carat = carat
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def getDFForData(self):
        try:
            data = {
                "carat": [self.carat],
                "depth": [self.depth],
                "table": [self.table],
                "x": [self.x],
                "y": [self.y],
                "z": [self.z],
                "cut": [self.cut],
                "color": [self.color],
                "clarity": [self.clarity],
            }
            df = pd.DataFrame(data)
            logging.info("Dataframe Gathered")
            return df
        except Exception as e:
            logging.info("Exception Occured in prediction pipeline")
            raise CustomException(e, sys)
