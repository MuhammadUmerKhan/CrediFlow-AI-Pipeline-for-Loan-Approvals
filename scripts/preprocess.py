import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from data_injestion import load_data
import logging
import os
import joblib
from config import PREPROCESSING_CONFIG, LOG_DIR, SCALER_PATH, PROCESSED_DATA_DIR

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

def preprocess_data():
    """Preprocess train and test datasets."""
    try:
        # Load data
        train_data, test_data = load_data()
        
        # Map categorical variables
        logging.info("⏳ Mapping categorical variables")
        train_data['cb_person_default_on_file'] = train_data['cb_person_default_on_file'].map({'Y': 1, 'N': 0}).astype(int)
        test_data['cb_person_default_on_file'] = test_data['cb_person_default_on_file'].map({'Y': 1, 'N': 0}).astype(int)
        
        loan_grade_mapping = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
        train_data['loan_grade'] = train_data['loan_grade'].map(loan_grade_mapping)
        test_data['loan_grade'] = test_data['loan_grade'].map(loan_grade_mapping)
        
        # One-hot encoding
        logging.info("⏳ Applying one-hot encoding")
        train_data = pd.get_dummies(train_data, drop_first=True).astype(float)
        test_data = pd.get_dummies(test_data, drop_first=True).astype(float)
        
        # Log transformation for numerical columns
        num_cols = PREPROCESSING_CONFIG['numerical_columns']
        
        logging.info(f"⏳ Applying log transformation to {num_cols}")
        for col in num_cols:
            if col in train_data.columns:
                train_data[col] = np.log1p(train_data[col])
                test_data[col] = np.log1p(test_data[col])
        
        # Split features and target
        X = train_data.drop(PREPROCESSING_CONFIG['target_column'], axis=1)
        y = train_data[PREPROCESSING_CONFIG['target_column']]
        
        # Apply SMOTEENN
        logging.info("⏳ Applying SMOTEENN for data balancing")
        smote_enn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        
        # Train-validation split
        logging.info("⏳ Splitting data into train and test sets")
        X_train, X_val, y_train, y_val = train_test_split(
            X_resampled, y_resampled, test_size=0.1, random_state=42, stratify=y_resampled
        )
        
        # Scale numerical columns
        logging.info("⏳ Scaling numerical columns")
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        test_data_scaled = test_data.copy()
        
        X_train_scaled[num_cols] = scaler.fit_transform(X_train_scaled[num_cols])
        X_val_scaled[num_cols] = scaler.transform(X_val_scaled[num_cols])
        test_data_scaled[num_cols] = scaler.transform(test_data_scaled[num_cols])
                
        # Save preprocessed data
        logging.info(f"⏳ Saving preprocessed data to {PROCESSED_DATA_DIR}")
        X_train_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_train_scaled.csv"), index=False)
        X_val_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, "X_val_scaled.csv"), index=False)
        y_train.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_train.csv"), index=False)
        y_val.to_csv(os.path.join(PROCESSED_DATA_DIR, "y_val.csv"), index=False)
        test_data_scaled.to_csv(os.path.join(PROCESSED_DATA_DIR, "test_data_scaled.csv"), index=False)
        
        logging.info(f"Train size: {X_train_scaled.shape}")
        logging.info(f"Test size: {test_data_scaled.shape}")
        
        logging.info("✅ Scaler Saved")
        joblib.dump(scaler, f"{SCALER_PATH}/scaler.pkl")
        
        logging.info("✅ Preprocessing completed successfully")
        return X_train_scaled, X_val_scaled, y_train, y_val, test_data_scaled
    
    except Exception as e:
        logging.error(f"⛔ Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    X_train_scaled, X_val_scaled, y_train, y_val, test_data_scaled = preprocess_data()