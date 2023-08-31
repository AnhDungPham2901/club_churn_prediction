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

def load_full_train_set():
    train = pd.read_csv(paths.PROCESSED_DATA + "/full_train_without_embed.csv")
    return train

def build_pytorch_tabular_model():
    df = load_full_train_set()
    train, test = train_test_split(df, random_state=42, test_size=0.15)
    train, val = train_test_split(train, random_state=42, test_size=0.15)
    num_col_names = columns.CONTINUOUS_COLUMNS
    categorical_columns = columns.CATEGORICAL_COLUMNS
    data_config = DataConfig(
        target=[
        "TARGET"
    ], 
        continuous_cols=num_col_names,
        categorical_cols=categorical_columns
    )
    trainer_config = TrainerConfig(
        auto_lr_find=True,
        batch_size=64,
        max_epochs=200,
    )
    optimizer_config = OptimizerConfig()

    model_config = CategoryEmbeddingModelConfig(
        task="classification",
        layers="1024-512-512",
        activation="LeakyReLU",
        learning_rate=1e-3,
        metrics = ['accuracy']
    )

    tabular_model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )
    tabular_model.fit(train=train, validation=val)
    result = tabular_model.evaluate(test)
    pred_df = tabular_model.predict(test)
    tabular_model.save_model(paths.MODEL_EMBEDED)
    return tabular_model