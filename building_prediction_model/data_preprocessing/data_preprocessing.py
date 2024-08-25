import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pathlib import Path
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent.parent
sys.path.append(str(PACKAGE_ROOT))

from building_prediction_model.config import config

# Transformer to drop specified columns
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, variables_to_drop=None):
        self.variables_to_drop = variables_to_drop or config.FEATURES_DROP
    
<<<<<<<<<<<<<<  ✨ Codeium Command 🌟  >>>>>>>>>>>>>>>>
    def fit(self, X, y=None):
        """
        Fit simply returns self.
        
        Parameters
        ----------
        X : pandas.DataFrame
            The features to be transformed.
        y : pandas.Series, optional
            The target variable, by default None.
        """
        return self
<<<<<<<  8e29ce0e-1e78-46e7-ad01-239790fb1e65  >>>>>>>
    
    def transform(self, X):
        X = X.drop(columns=self.variables_to_drop)
        return X

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
    
