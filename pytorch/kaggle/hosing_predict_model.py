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

iowa_file_path = '../kaggle/input/home-data-for-ml-course/train.csv'
home_data = pd.read_csv(iowa_file_path)
X_full = pd.read_csv(iowa_file_path, index_col='Id')
X_test_full = pd.read_csv(iowa_file_path, index_col='Id')
X_full.head()
X_test_full.head()