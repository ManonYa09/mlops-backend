import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # Adjusting to go up two directories to the project root
sys.path.append(str(PACKAGE_ROOT))

# Now, import the config module from building_prediction_model
from building_prediction_model.config import config

print(config.FEATURES_DROP)
# Transformer to drop specified columns
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop or config.FEATURES_DROP
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.drop(columns=self.variables_to_drop)
        return X

        return self
# Transformer to encode and create dummy variables
class EncodeAndBind(BaseEstimator, TransformerMixin):
    def __init__(self, encode=None, dummy=None):
        self.encode = encode
        self.dummy = dummy
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X[self.encode].replace({'Male': 0, 'Female': 1}, inplace=True)
        X = pd.get_dummies(X, columns=[self.dummy])
        X.replace({True: 1, False: 0}, inplace=True)
        return X

# Transformer to normalize specified variables
class Scale(BaseEstimator, TransformerMixin):
    def __init__(self, variables=None):
        self.variables = variables or config.FEATURES_TO_SCALE
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = (X[variable] - X[variable].min()) / (X[variable].max() - X[variable].min())
        return X
    