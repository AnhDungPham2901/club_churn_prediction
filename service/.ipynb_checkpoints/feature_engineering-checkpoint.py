import sys
import numpy as np
import pandas as pd
from datetime import datetime
sys.path.append('/home/dungpa/club_churn_predictionn/')
from configuration import paths, columns
from sklearn.model_selection import train_test_split
from pytorch_tabular import TabularModel
from pytorch_tabular.models import CategoryEmbeddingModelConfig
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
from pytorch_tabular.categorical_encoders import CategoricalEmbeddingTransformer


def load_train_raw():
    df = pd.read_excel(paths.RAW_DATA + "/club_churn_train.xlsx")
    return df

def load_test_raw():
    df = pd.read_excel(paths.RAW_DATA + "/club_churn_test.xlsx")
    return df

def create_agent_performance(df):
    agent_customer_counts = df['AGENT_CODE'].value_counts()
    updated_agent_performance_groups = agent_customer_counts.map(lambda x: 0 if x == 1 else
                                                                1 if 1 < x <= 5 else
                                                                2 if 5 < x <= 15 else
                                                                3)
    df['AGENT_PERFORMANCE'] = df['AGENT_CODE'].map(updated_agent_performance_groups)
    return df

def fill_nan(df):
    df['MEMBER_MARITAL_STATUS'].fillna('Unknown', inplace=True)
    df['MEMBER_GENDER'].fillna('Unknown', inplace=True)

    # For numerical variables
    df['MEMBER_ANNUAL_INCOME'].fillna(df['MEMBER_ANNUAL_INCOME'].median(), inplace=True)
    df['MEMBER_OCCUPATION_CD'].fillna(df['MEMBER_OCCUPATION_CD'].median(), inplace=True)
    return df

def create_income_to_fee_ratio(df):
    df['INCOME_TO_FEE_RATIO'] = np.divide(df['ANNUAL_FEES'], df['MEMBER_ANNUAL_INCOME'])
    # Replace inf or -inf with NaN or a desired value
    df['INCOME_TO_FEE_RATIO'].replace([np.inf, -np.inf], 0, inplace=True)
    return df

def create_date_features(df):
    # Convert START_DATE and END_DATE to datetime format for feature extraction
    df['START_DATE'] = pd.to_datetime(df['START_DATE'], format='%Y%m%d', errors='coerce')
    current_date = datetime(2014,1,1)
    df['DAYS_STAY'] = (current_date - df['START_DATE']).dt.days
    # Extract year, month, and day from START_DATE and END_DATE
    df['START_YEAR'] = df['START_DATE'].dt.year
    df['START_MONTH'] = df['START_DATE'].dt.month
    df['START_DAY'] = df['START_DATE'].dt.day
    return df

def create_start_year_age_combined(df):
    df['START_YEAR_AGE_COMBINED'] = df['START_YEAR'] * df["MEMBER_AGE_AT_ISSUE"]
    return df

def create_train_master_df():
    df = load_train_raw()
    df = create_agent_performance(df)
    df = fill_nan(df)
    df = create_income_to_fee_ratio(df)
    df = create_date_features(df)
    df = create_start_year_age_combined(df)
    
    # Encoding the target variable
    df['TARGET'] = df['MEMBERSHIP_STATUS'].apply(lambda x: 1 if x == 'CANCELLED' else 0)
    df.drop(columns=columns.DROP_COLUMNS_TRAIN, inplace=True)
    df.to_csv(paths.PROCESSED_DATA + "/full_train_without_embed.csv", index=False)
    return df

def create_test_master_df():
    df = load_test_raw()
    df = create_agent_performance(df)
    df = fill_nan(df)
    df = create_income_to_fee_ratio(df)
    df = create_date_features(df)
    df = create_start_year_age_combined(df)    
    df.drop(columns=columns.DROP_COLUMNS_TEST, inplace=True)
    return df

def load_tabular_model():
    tabular_model = TabularModel.load_from_checkpoint(paths.MODEL_EMBEDED)
    return tabular_model
    
def load_full_train_set():
    train = pd.read_csv(paths.PROCESSED_DATA + "/full_train_without_embed.csv")
    return train
    
def build_pytorch_categorical_transformer():
    tabular_model = load_tabular_model()
    train = load_full_train_set()
    transformer = CategoricalEmbeddingTransformer(tabular_model)
    train_transformed = transformer.fit_transform(train)
    return transformer

def embed_categorical_features(df, transformer):
    df = transformer.transform(df)
    print(df.shape)
    return df
