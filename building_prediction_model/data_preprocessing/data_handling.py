import pandas as pd
import os 
import joblib 
from pathlib import Path
import sys
from sklearn.model_selection import train_test_split

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
print(f"PACKAGE_ROOT: {PACKAGE_ROOT}")  # Debug print to check the resolved path
sys.path.append(str(PACKAGE_ROOT))

try:
    from building_prediction_model.config import config
except ImportError as e:
    print(f"Error importing config module: {e}")
    sys.exit(1)

print(PACKAGE_ROOT)

def load_dataset(file_name):
    filepath = os.path.join(config.DATAPATH,file_name)
    _data = pd.read_csv(filepath)
    _data.columns = [c.strip() for c in _data.columns] # Fix Column names
    return _data
def separate_data(data):
    X = data.drop(config.TARGET, axis=1)
    y= data[config.TARGET]
    return X,y

def split_data(X, y, test_size=0.2, random_state=42):
  # Split into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
  return X_train, X_test, y_train, y_test
# filepath = os.path.join(config.DATAPATH,'Churn_Modelling.csv')
# print(filepath )
# load_dataset('Churn_Modelling.csv')
print(load_dataset('Churn_Modelling.csv'))