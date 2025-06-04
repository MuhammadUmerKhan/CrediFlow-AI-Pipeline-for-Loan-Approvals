# 📋 AI-Driven Loan Approval Prediction System - Scripts Folder

Welcome to the `scripts` folder of the **AI-Driven Loan Approval Prediction System**! 🚀 This directory contains all the Python scripts that power the end-to-end machine learning pipeline, from data ingestion to model deployment. Below, you'll find a detailed overview of each script, their purpose, and how to use them. Let's dive in! 🌊

---

## 📖 Project Overview
This project leverages machine learning (specifically a TensorFlow-based Artificial Neural Network) to predict loan approval outcomes based on applicant data. The system includes data preprocessing, model training, evaluation, registration, and a Streamlit-based web interface for predictions. The scripts in this folder form the backbone of the ML pipeline. 🛠️

---

## 📂 Folder Structure
- `scripts/`: Contains all Python scripts for the ML pipeline.
- `data/`: Stores raw data files (`train.csv`, `test.csv`).
- `models/`: Holds saved models and scalers.
- `logs/`: Contains log files for tracking execution.
- `database/`: Stores the MLflow database (`mlflow.db`).
- `artifacts/`: Stores MLflow artifacts.

---

## 📋 Key Scripts and Their Purposes

### 1. 🌐 `data_injestion.py`
- **Purpose**: Loads raw train and test datasets from CSV files.
- **Key Features**:
  - Checks for file existence at `TRAIN_PATH` and `TEST_PATH`.
  - Uses pandas to read data with `id` as the index.
  - Logs progress and errors to `app.log`.
- **Usage**: Run directly to load data: `python data_injestion.py`.
- **Output**: Returns `train_data` and `test_data` DataFrames.

### 2. 🔧 `preprocess.py`
- **Purpose**: Preprocesses data for model training.
- **Key Features**:
  - Maps categorical variables (e.g., `cb_person_default_on_file`, `loan_grade`).
  - Applies one-hot encoding and log transformation.
  - Balances data with SMOTEENN and scales features using `StandardScaler`.
  - Saves preprocessed data (e.g., `X_train_scaled.csv`) and scaler to `models/`.
- **Usage**: Run directly: `python preprocess.py`.
- **Output**: Preprocessed datasets and `scaler.pkl`.

### 3. 🧠 `train.py`
- **Purpose**: Trains the TensorFlow ANN model.
- **Key Features**:
  - Builds a sequential model with configurable layers from `config.py`.
  - Trains on preprocessed data with early stopping support.
  - Logs metrics, hyperparameters, and history to MLflow.
  - Saves the model to `models/loan_approval_model.keras`.
- **Usage**: Run directly: `python train.py`.
- **Output**: Trained model and MLflow logs.

### 4. 📊 `evaluate.py`
- **Purpose**: Evaluates the model and generates test predictions.
- **Key Features**:
  - Loads preprocessed validation and test data.
  - Evaluates model performance (loss, accuracy, AUC, precision, recall).
  - Generates predictions and logs them to MLflow as `test_predictions.csv`.
- **Usage**: Run with: `python evaluate.py --model_name LoanApprovalModel --alias ReadyForProduction`.
- **Output**: Evaluation metrics and test predictions.

### 5. 📦 `register_model.py`
- **Purpose**: Registers the trained model in the MLflow Model Registry.
- **Key Features**:
  - Registers the model with a specified name and alias (e.g., `ReadyForProduction`).
  - Adds tags (e.g., dataset, model type) and a description.
  - Uses the latest run if no `run_id` is provided.
- **Usage**: Run with: `python register_model.py`.
- **Output**: Registered model version.

### 6. 🎯 `pipeline.py`
- **Purpose**: Orchestrates the entire ML pipeline.
- **Key Features**:
  - Executes preprocessing, training, registration, and evaluation sequentially.
  - Logs each step and handles exceptions.
  - Returns a dictionary with model details and metrics.
- **Usage**: Run directly: `python pipeline.py`.
- **Output**: Pipeline results including model version and evaluation metrics.

### 7. 🛠️ `utils.py`
- **Purpose**: Provides utility functions for data handling.
- **Key Features**:
  - Loads preprocessed data files from `PROCESSED_DATA_DIR`.
  - Returns scaled train, validation, and test datasets.
- **Usage**: Imported by other scripts (e.g., `train.py`, `evaluate.py`).
- **Output**: DataFrames and arrays for model training.

### 8. ⚙️ `config.py`
- **Purpose**: Centralizes project configuration settings.
- **Key Features**:
  - Defines file paths (e.g., `TRAIN_PATH`, `MODEL_DIR`).
  - Specifies preprocessing, model, and training parameters.
  - Sets logging and MLflow directories.
- **Usage**: Imported by all scripts for consistent settings.

---

## 🚀 How to Run the Project

1. **Prerequisites**:
   - Install dependencies: `pip install -r requirements.txt` (ensure `requirements.txt` includes `tensorflow`, `pandas`, `numpy`, `scikit-learn`, `mlflow`, `imbalanced-learn`, `joblib`).
   - Set up the directory structure with `data/`, `models/`, `logs/`, and `database/` folders.
   - Place `train.csv` and `test.csv` in the `data/` folder.

2. **Run the Pipeline**:
   - Execute the full pipeline: `python scripts/pipeline.py`.
   - This will preprocess data, train the model, register it, and evaluate it.

3. **Individual Scripts**:
   - Run specific steps (e.g., `python scripts/train.py` or `python scripts/evaluate.py`).

4. **Streamlit App**:
   - After training and registration, run the app: `streamlit run scripts/loan_predictor.py` (assuming `loan_predictor.py` is in this folder).

---

## 📝 Logging
- All scripts log progress and errors to `logs/app.log`.
- Use `tail -f logs/app.log` to monitor in real-time.