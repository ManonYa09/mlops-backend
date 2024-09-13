from sklearn.pipeline import Pipeline
from pathlib import Path
import os
import sys
import warnings 
warnings.filterwarnings("ignore")
PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from sklearn.linear_model import LogisticRegression

from building_prediction_model.config import config
import building_prediction_model.data_preprocessing.data_preprocessing as pp
import numpy as np 
# print(config.DATASET)
classification_pipeline = Pipeline(
    steps =  [(
    'Drop_columns', pp.DropColumns(variables_to_drop =['RowNumber','CustomerId','Surname'])),
    ('Encode_and_bind', pp.EncodeAndBind(encode='Gender', dummy='Geography')),
    ('Scale', pp.Scale(variables=['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary'])),
    ('Model', LogisticRegression(random_state=12))])


print(classification_pipeline)