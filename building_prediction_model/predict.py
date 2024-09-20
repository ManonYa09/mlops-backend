import pandas as pd
import numpy as np
import joblib
import warnings 
warnings.filterwarnings("ignore")
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from building_prediction_model.config import config
from building_prediction_model.data_preprocessing.data_handling import load_dataset, load_pipeline, separate_data

config.SAVE_MODEL_PATH
# Load the saved classification pipeline
# classification_pipeline = load_pipeline('classification.pkl')
def laod_pipeline(MODEL_NAME):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    print(save_path)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded
classification_pipeline  = joblib.load('/Users/macbook/Desktop/MLOPS/ml-prediction-backend/building_prediction_model/trained_models/classification.pkl')
# print(classification_pipeline)

def generate_predictions():
    test_data = load_dataset(config.TEST_FILE)
    X, y = separate_data(test_data)
    # X = test_data.drop('Exited', axis=1)  # Features
    # y = test_data['Exited']  # Actual target variable (Exited)
    
    # Predict using the loaded classification pipeline
    pred = classification_pipeline.predict(X)
    pred1 = load_pipeline('classification.pkl').predict(X)
    # pred2 = load_pipeline('classification.pkl').predict([1236,15600700,'Pan',523,'Germany','Male',63,6,116227.27,1,1,1,119404.63])
    print(X)
    # Convert the predictions to readable output
    output = np.where(pred == 1, 'Churn', 'Not Churn')
    pred1 = np.where(pred1 == 1, 'Churn11', 'Not Churn11')
    
    # Print or return the predicted outputs
    print("Predicted output: ", pred1, len(pred1), X)
    return  pred1

if __name__ == '__main__':
    generate_predictions()