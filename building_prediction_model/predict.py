import pandas as pd
import numpy as np
import joblib

from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from building_prediction_model.config import config
from building_prediction_model.data_preprocessing.data_handling import load_dataset, load_pipeline, separate_data


# joblib.load(os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME))
classification_pipeline = joblib.load('/Users/macbook/Desktop/MLOPS/ml-prediction-backend/building_prediction_model/trained_models/classification.pkl')

# classification_pipeline = load_pipeline(config.MODEL_NAME)




def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    X,y = separate_data(test_data)

    pred = classification_pipeline.predict(X)
    output = np.where(pred==1,'Churn','Not Churn')
    print(output)
    return output

if __name__=='__main__':
    generate_predictions()
