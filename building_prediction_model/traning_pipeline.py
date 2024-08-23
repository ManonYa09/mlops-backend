import pandas as pd
import numpy as np 
from pathlib import Path
import os
import sys
import joblib

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

# # Then perform import
from building_prediction_model.config import config  
from building_prediction_model.data_preprocessing.data_handling import load_dataset,separate_data,split_data,save_pipeline
import building_prediction_model.data_preprocessing.data_preprocessing as pp 
import building_prediction_model.pipeline as pipe 
import sys



# Define the save directory and file name
save_dir = '/Users/macbook/Desktop/MLOPS/ml-prediction-backend/building_prediction_model/trained_models/'
save_path = os.path.join(save_dir, 'classification.pkl')

# # Check if the directory exists
# if not os.path.exists(save_dir):
#     print(f"Directory does not exist: {save_dir}")
# else:
#     print(f"Directory exists: {save_dir}")

#Save the pipeline
try:
    joblib.dump(pipe.classification_pipeline, save_path)
    print(f"Model has been saved to {save_path}")
except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
# print('I love You')
# print(config.DATAPATH)

def perform_pipeline():
    dataset = load_dataset('Churn_Modelling.csv')
    X,y = separate_data(dataset)
    X_train, X_test, y_train, y_test = split_data(X,y)
    test_data = X_test.copy()
    test_data[config.TARGET] = y_test
    test_data.to_csv(os.path.join(config.DATAPATH,config.TEST_FILE))
    pipe.classification_pipeline.fit(X_train,y_train)

#Save the pipeline
    try:
        joblib.dump(pipe.classification_pipeline, save_path)
        print(f"Model has been saved to {save_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    # save_pipeline(pipe.classification_pipeline)
if __name__=='__main__':
    perform_pipeline()
