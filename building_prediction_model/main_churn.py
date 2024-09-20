from fastapi import FastAPI
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from starlette.middleware.cors import CORSMiddleware
from trained_models.PredictionModel_Churn import PredictionModel
# from service.PredictionService import *
import joblib
import os

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))
from building_prediction_model.config import config
from building_prediction_model.config import config
from building_prediction_model.data_preprocessing.data_handling import load_dataset, load_pipeline, separate_data
def laod_pipeline(MODEL_NAME):
    save_path = os.path.join(config.SAVE_MODEL_PATH,config.MODEL_NAME)
    print(save_path)
    model_loaded = joblib.load(save_path)
    print(f"Model has been loaded")
    return model_loaded

test_data = load_dataset(config.TEST_FILE)
X, y = separate_data(test_data)
    # X = test_data.drop('Exited', axis=1)  # Features
    # y = test_data['Exited']  # Actual target variable (Exited)
    
    # Predict using the loaded classification pipeline
    # pred = classification_pipeline.predict(X)
pred1 = load_pipeline('classification.pkl').predict(X)
    # pred2 = load_pipeline('classification.pkl').predict([1236,15600700,'Pan',523,'Germany','Male',63,6,116227.27,1,1,1,119404.63])
    # Convert the predictions to readable output
# output = np.where(pred == 1, 'Churn', 'Not Churn')
pred1 = np.where(pred1 == 1, 'Churn11', 'Not Churn11')
    
    # Print or return the predicted outputs
print("Predicted output: ", pred1, len(pred1))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


input_data = {
        'RowNumber':[1236, 1, 1],
        'CustomerId': [15600700, 1, 1],
        'Surname': ['Pan', '1','a'],
        'CreditScore':[523, 500, 500],
        'Geography': ['Spain', 'France', 'Germany'],
        'Gender': ['Male', 'Female', 'Male'],
        'Age': [63, 50, 40],
        'Tenure': [6, 1, 2],
        'Balance':[ 116227.27, 2, 3],
        'NumOfProducts': [1, 1, 1],
        'HasCrCard':[ 1, 1, 1],
        'IsActiveMember':[1, 1, 1],
        'EstimatedSalary':[119404.63, 3, 3 ]
    }
input_data = pd.DataFrame(input_data)
print(input_data.head(1))
print(X.head(1))
print(f'this is :{load_pipeline("classification.pkl").predict(X)}')
# print(predict11)
# num = np.float32(predict11)

# result = 'Churn' if float(num) > 0.5 else 'Non Churn'
# print(result)
# # @app.post("/prediction_churn/")
# # async def say_hello(request: PredictionModel):
# #     input_data = {
# #         'RowNumber': request.RowNumber,
# #         'CustomerId': request.CustomerId,
# #         'Surnam': request.Surnam,
# #         'CreditScore': request.CreditScore,
# #         'Geography': request.Geography,
# #         'Gender': request.Gender,
# #         'Age': request.Age,
# #         'Tenure': request.Tenure,
# #         'Balance': request.Balance,
# #         'NumOfProducts': request.NumOfProducts,
# #         'HasCrCard': request.HasCrCard,
# #         'IsActiveMember': request.IsActiveMember,
# #         'EstimatedSalary': request.EstimatedSalary
# #     }
# #     predict = predict_churn.predict(pd.DataFrame(input_data))
# #     num = np.float32(predict)

# #     result = 'Churn' if float(num) > 0.5 else 'Non Churn'
# #     return {
# #         "probability": float(num),
# #         "prediction": result
# #     }

