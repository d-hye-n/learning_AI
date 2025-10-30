import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
pd.set_option('display.max_columns', None)

train_file_path = '../kaggle/input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(train_file_path)
X_full = pd.read_csv(train_file_path, index_col='Id')
X_test_full = pd.read_csv(train_file_path, index_col='Id')
X_full.head()
X_test_full.head()

# 1. Separate target from predictors
y = X_full['SalePrice'] #defining target variable
X_full.drop('SalePrice', axis=1, inplace=True) #dropping target variable from features

# 2. Separete categorical and numerical columns
categorical_cols = [col for col in X_full.columns if
                    (X_full[col].dtype == "object")]
numerical_cols = [col for col in X_full.columns if
                  X_full[col].dtype in ['int64', 'float64'] and
                  col != "GarageYrBlt"]

