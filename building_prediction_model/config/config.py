import os
import sys
import pathlib as Path
# Adding the parent directory to the sys.path

PACKAGE_ROOT = Path.Path(os.path.abspath(os.path.dirname(__file__))).parent.parent

# Append PACKAGE_ROOT to sys.path so Python can find building_prediction_model
sys.path.append(str(PACKAGE_ROOT))

DATAPATH = os.path.join(PACKAGE_ROOT, "datasets")

TARGET = 'Exited'

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

print(PACKAGE_ROOT)
print(DATAPATH)