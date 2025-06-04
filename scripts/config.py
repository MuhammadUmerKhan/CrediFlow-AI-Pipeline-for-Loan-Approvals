# Configuration settings for the loan prediction project
import os

# Data paths
TRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "train.csv"))
TEST_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test.csv"))
PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

# Preprocessing parameters
PREPROCESSING_CONFIG = {
    'numerical_columns': ['loan_amnt', 'loan_int_rate', 'person_income', 'person_age', 'person_emp_length'],
    'categorical_columns': ['loan_grade', 'cb_person_default_on_file', 'person_home_ownership', 'loan_intent'],
    'target_column': 'loan_status'
}

# Model parameters
MODEL_LAYERS = [
    {'units': 128, 'activation': 'relu', 'dropout': 0.2},
    {'units': 64, 'activation': 'relu', 'dropout': 0.2},
    {'units': 32, 'activation': 'relu', 'dropout': 0.2},
    {'units': 16, 'activation': 'relu'}
]
OUTPUT_ACTIVATION = 'sigmoid'
OPTIMIZER = 'adam'
LOSS = 'binary_crossentropy'
METRICS = ['accuracy']
EPOCHS = 20
BATCH_SIZE = 64
EARLY_STOPPING_PATIENCE = 5

# File paths
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts"))

SCALER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
MLFLOW_DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "mlflow.db"))
MLFLOW_ARTIFACT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))