import pandas as pd
import logging
import os
from config import LOG_DIR, PROCESSED_DATA_DIR

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

def load_preprocessed_data():
    """Read all preprocessed data from PROCESSED_DATA_DIR."""
    try:
        logging.info(f"⏳ Reading preprocessed data from {PROCESSED_DATA_DIR}")
        X_train_scaled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_train_scaled.csv"))
        X_val_scaled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "X_val_scaled.csv"))
        y_train = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv")).values.ravel()
        y_val = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "y_val.csv")).values.ravel()
        test_data_scaled = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "test_data_scaled.csv"))
        
        logging.info(f"Loaded X_train_scaled: {X_train_scaled.shape}")
        logging.info(f"Loaded X_val_scaled: {X_val_scaled.shape}")
        logging.info(f"Loaded y_train: {y_train.shape}")
        logging.info(f"Loaded y_val: {y_val.shape}")
        logging.info(f"Loaded test_data_scaled: {test_data_scaled.shape}")
        
        logging.info("✅ Preprocessed data loaded successfully")
        return X_train_scaled, X_val_scaled, y_train, y_val, test_data_scaled
    
    except Exception as e:
        logging.error(f"⛔ Error reading preprocessed data: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        X_train_scaled, X_val_scaled, y_train, y_val, test_data_scaled = load_preprocessed_data()
        logging.info("✅ Data reading completed successfully")
    except Exception as e:
        logging.error(f"⛔ Error in main: {str(e)}")
        raise