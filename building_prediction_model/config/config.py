import os
import sys
import pathlib as Path
# Adding the parent directory to the sys.path

PACKAGE_ROOT = Path.Path(os.path.abspath(os.path.dirname(__file__))).parent

# Append PACKAGE_ROOT to sys.path so Python can find building_prediction_model
sys.path.append(str(PACKAGE_ROOT))
print(PACKAGE_ROOT)
DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

DATASET = 'Churn_Modelling.csv'
TARGET = 'Exited'
TEST_FILE = 'test_data.csv'

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,'trained_models')

### Final fearues used in the model
FEATURES = ['CreditScore',
             'Geography', 
             'Gender',
               'Age',
                 'Tenure',
                   'Balance', 
                   'NumOfProducts',
                     'HasCrCard', 
                     'IsActiveMember',
                       'EstimatedSalary']
PRE_FEATURES = ['RowNumber','CustomerId','Surname']
NUM_FEATUES = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
CAT_FEATUES = ['Geography','Gender','HasCrCard','IsActiveMember']

FEATURES_TO_ENCODE = ['Geography','Gender']
FEATURES_TO_SCALE = ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']

FEATURES_DROP = ['RowNumber','CustomerId','Surname']

# print(PACKAGE_ROOT)
# print(DATAPATH)
print(SAVE_MODEL_PATH)