import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import BorderlineSMOTE
from collections import Counter

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', 20)

train = pd.read_csv("train.csv")
# test = pd.read_csv("test.csv")
val = pd.read_csv("val.csv")

def orange_MONTHS(age):
  if age <= 8:
    return '1'
  elif age <= 15:
    return '2'
  elif age <= 25:
    return '3'
  elif age <= 40:
    return '4'
  else:
    return '5'
  

def speed_group(speed):
  if speed <= 60:
    return '1'
  else:
    return '2'      
  
def age_group(age):
  if age <= 23:
    return '1'
  elif age <= 25:
    return '2'
  elif age <= 30:
    return '3'
  elif age <= 35:
    return '4'
  elif age <= 40:
    return '5'
  elif age <= 45:
    return '6'
  elif age <= 50:
    return '7'
  elif age <= 55:
    return '8'
  else:
    return '9'        



def preprocessing(data):
  
  # Remove unnecessary columns
  remove_columns = ['OF_PREV_SPEED', 'LAST_LINK_PRIORITY', 'LAST_POWER_VALIDATION', 'LAST_LINK_STATUS', 'LAST_LINK_QUALITY',
                    'Disconnection_TOTAL_MIN_day', 'ID', 'Disconnection_TOTAL_MAX_day', 'Disconnection_TOTAL_SUM_Month',
                    'Disconnection_TOTAL_MEAN_Month']
  data = data.drop(remove_columns, axis=1)


  # Remove rows with lots of missing data
  row_missing_percentage = .25

  data = data.loc[(data.isna().sum(1) / data.shape[1]) <= row_missing_percentage, :]
  data = data.reset_index(drop=True)

  assert not ((data.isna().sum(1) / data.shape[1]) > .25).any(), 'Error'


  # Fill missing data
  categorical_columns = data.select_dtypes('O').columns
  numerical_columns = data.select_dtypes(np.number).columns


  # Categorical data
  governorates_missing = 'West Amman'
  customer_gender = 'M'

  data['GOVERNORATE'].fillna(value=governorates_missing, inplace=True)
  data['CUSTOMER_GENDER'].fillna(value=customer_gender, inplace=True)

  assert not data[categorical_columns].isna().any().any(), 'Error'


  # Numerical data
  ### Median
  numerical_missing_values = {'GB_TOTAL_CONSUMPTION_Month1': 361.79541484406195,
                              'GB_TOTAL_CONSUMPTION_Month2': 354.923208067659,
                              'GB_TOTAL_CONSUMPTION_Month3': 333.67581848939847}

  for num_miss_value in numerical_missing_values.keys():
    data[num_miss_value].fillna(value=numerical_missing_values[num_miss_value], inplace=True)

  assert not data[num_miss_value].isna().any().any(), 'Error'
  

  # Feature engineering
  data = data[data['GOVERNORATE'].isin(['West Amman', 'East Amman'])]

  data.loc[~data['GOVERNORATE'].isin(['West Amman', 'East Amman']), 'GOVERNORATE'] = 'OTHER'
  data.loc[data['CUSTOMER_GENDER'] == 'U', 'CUSTOMER_GENDER'] = 'M'

  data['CUSTOMER_AGE_YEARS'] = data['CUSTOMER_AGE_MONTHS'] // 12
  data = data[(data['CUSTOMER_AGE_YEARS'] >= 18) & (data['CUSTOMER_AGE_YEARS'] <= 60)]
  data['CUSTOMER_AGE_YEARS'] = data['CUSTOMER_AGE_YEARS'].apply(age_group)
  data['AGE_GENDER'] = data['CUSTOMER_AGE_YEARS'] + data['CUSTOMER_GENDER']
  data.drop(['CUSTOMER_AGE_MONTHS', 'CUSTOMER_AGE_YEARS', 'CUSTOMER_GENDER'], axis=1, inplace=True)

  data['OF_SPEED'] = data['OF_SPEED'].apply(speed_group)

  # data['Customer with orange_MONTHS'] = data['Customer with orange_MONTHS'].apply(orange_MONTHS)

  data['GB_TOTAL_CONSUMPTION'] = (data['GB_TOTAL_CONSUMPTION_Month1'] +
                                  data['GB_TOTAL_CONSUMPTION_Month2'] +
                                  data['GB_TOTAL_CONSUMPTION_Month3']) / 3
  data.drop(['GB_TOTAL_CONSUMPTION_Month1', 'GB_TOTAL_CONSUMPTION_Month2', 'GB_TOTAL_CONSUMPTION_Month3'], axis=1, inplace=True)

  # Encode categorical data
  categorical_columns = ['GOVERNORATE', 'MIGRATION_FLAG', 'AGE_GENDER', 'OF_SPEED']#, 'Customer with orange_MONTHS']
  data = pd.get_dummies(data, columns=categorical_columns)

  COMMITMENT_MAP = {36: 3, 24: 2, 12: 1}
  data['COMMITMENT'] = data['COMMITMENT'].map(COMMITMENT_MAP)

  # Scale the data
  ### RobustScaler = (Xi-Xmedian) / Xiqr

  data_median = {'GB_TOTAL_CONSUMPTION': 375}
  data_iqr = {'GB_TOTAL_CONSUMPTION': 347}


  for key in data_median.keys():
    data[key] = (data[key] - data_median[key]) / data_iqr[key]

  return data


train = preprocessing(train)
val = preprocessing(val)


X_train, y_train = train.drop('TARGET', axis=1), train['TARGET']
X_val, y_val = val.drop('TARGET', axis=1), val['TARGET']

oversample = BorderlineSMOTE(sampling_strategy=.1)
X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)



def evaluation(y_true, y_pred):
  precision = precision_score(y_true, y_pred, average='macro')
  recall = recall_score(y_true, y_pred, average='macro')
  f1 = f1_score(y_true, y_pred, average='macro')
  return {"precision macro":precision,"recall macro":recall,"F1_Score_macro":f1}


model = XGBClassifier(max_depth=3, scale_pos_weight=400, random_state=42)


model.fit(X_train_over, y_train_over)

train_prediction = model.predict(X_train)
val_prediction = model.predict(X_val)


import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing, f)