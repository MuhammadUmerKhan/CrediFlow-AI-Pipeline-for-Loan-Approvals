import pandas as pd
import os
import logging
from config import TRAIN_PATH, TEST_PATH, LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

def load_data():
    """Load train and test datasets from CSV files."""
    try:
        # ✅ Check if file exists
        if not os.path.exists(TRAIN_PATH) or not os.path.exists(TEST_PATH):
            logging.error("⛔ Data files not found at specified paths.")
            raise FileNotFoundError("⛔ Train or test CSV file not found.")

        print(f"📂 Loading dataset from: {TRAIN_PATH}")
        logging.info(f"⏳ Loading train data from {TRAIN_PATH}")
        train_data = pd.read_csv(TRAIN_PATH, index_col='id')
        
        logging.info(f"⏳ Loading test data from {TEST_PATH}")
        test_data = pd.read_csv(TEST_PATH, index_col='id')
        
        logging.info("✅ Data loaded successfully.")
        print("✅ Data Loaded Successfully! Shape:", train_data.shape)
        return train_data, test_data

    except Exception as e:
        logging.error(f"⛔ Error in data ingestion: {str(e)}")
        raise

if __name__ == "__main__":
    train_data, test_data = load_data()