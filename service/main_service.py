import pickle
import sys
import numpy as np
import pandas as pd
from datetime import datetime
sys.path.append('/home/dungpa/club_churn_predictionn/')
from configuration import paths, columns
from service import feature_engineering

def load_pretrained_model():
    model_path = paths.MODEL_RF + "/rf_model"
    with open(model_path, 'rb') as model_file:
        loaded_rf_model = pickle.load(model_file)
    return loaded_rf_model
    
def prepare_data():
    df = feature_engineering.create_test_master_df()
    transformer = feature_engineering.build_pytorch_categorical_transformer()
    df = feature_engineering.embed_categorical_features(df, transformer)
    return df

def return_predict():
    rf_model = load_pretrained_model()
    df = prepare_data()
    df['MEMBERSHIP_STATUS'] = rf_model.predict(df)
    df['MEMBERSHIP_STATUS']  = df['MEMBERSHIP_STATUS']\
    .map(lambda x: 'CANCELLED' if x == 1 else 'INFORCE')
    df = df[['MEMBERSHIP_STATUS']]
    raw_test = feature_engineering.load_test_raw()
    raw_test = raw_test.drop('MEMBERSHIP_STATUS', axis=1)
    result = raw_test.join(df)
    result.to_excel(paths.PROCESSED_DATA + "/club_churn_test_result.xlsx", index=False)
    return result