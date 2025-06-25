import tensorflow as tf
import mlflow
import mlflow.tensorflow
import pandas as pd
import logging
import os
import json
from utils import load_preprocessed_data
from config import MODEL_LAYERS, OUTPUT_ACTIVATION, OPTIMIZER, \
    LOSS, METRICS, EPOCHS, BATCH_SIZE, MODEL_DIR, LOG_DIR, MLFLOW_DB_PATH

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

def build_model(input_shape):
    """Build the neural network model."""
    try:
        model = tf.keras.Sequential()
        for layer in MODEL_LAYERS:
            model.add(tf.keras.layers.Dense(
                units=layer['units'],
                activation=layer['activation'],
                input_shape=(input_shape,)
            ))
            if 'dropout' in layer:
                model.add(tf.keras.layers.Dropout(layer['dropout']))
        
        model.add(tf.keras.layers.Dense(1, activation=OUTPUT_ACTIVATION))
        
        model.compile(
            optimizer=OPTIMIZER,
            loss=LOSS,
            metrics=METRICS
        )
        return model
    except Exception as e:
        logging.error(f"⛔ Error building model: {str(e)}")
        raise

def train_model():
    """Train the model and log with MLflow."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        
        # Load preprocessed data
        logging.info("⏳ Loading preprocessed data")
        X_train_scaled, X_val_scaled, y_train, y_val, test_data_scaled = load_preprocessed_data()
        
        # Build model
        logging.info("⏳ Building model")
        model = build_model(X_train_scaled.shape[1])
        
        # Log training parameters
        logging.info("Training parameters:")
        logging.info(f" ➤ Model Layers: {MODEL_LAYERS}")
        logging.info(f" ➤ Output Activation: {OUTPUT_ACTIVATION}")
        logging.info(f" ➤ Optimizer: {OPTIMIZER}")
        logging.info(f" ➤ Loss: {LOSS}")
        logging.info(f" ➤ Metrics: {METRICS}")
        logging.info(f" ➤ Epochs: {EPOCHS}")
        logging.info(f" ➤ Batch Size: {BATCH_SIZE}")
        
        # Prepare hyperparameters
        hyperparameters = {
            "model_layers": MODEL_LAYERS,
            "output_activation": OUTPUT_ACTIVATION,
            "optimizer": OPTIMIZER,
            "loss": LOSS,
            "metrics": METRICS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE
        }
        
        # Prepare dataset info
        dataset_info = {
            "dataset_name": "Loan Dataset",
            "train_shape": X_train_scaled.shape,
            "validation_shape": X_val_scaled.shape,
            "test_shape": test_data_scaled.shape,
            "num_features": X_train_scaled.shape[1],
            "target_classes": ["No", "Yes"]
        }
        
        # Set up MLflow
        mlflow.set_experiment("Loan_Prediction")
        with mlflow.start_run(run_name="model_training"):
            # Log parameters
            mlflow.log_params({
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'layers': len(MODEL_LAYERS),
                'optimizer': OPTIMIZER,
                'loss': LOSS
            })
            mlflow.set_tag("run_type", "training")
            mlflow.set_tag("dataset_used", "Loan Dataset")
            mlflow.set_tag("tracking_method", "SQLite Database")
            
            # Log hyperparameters as JSON
            logging.info("⏳ Logging hyperparameters")
            mlflow.log_text(json.dumps(hyperparameters, indent=4), "hyperparameters.json")
            
            # Log dataset info as JSON
            logging.info("⏳ Logging dataset info")
            mlflow.log_text(json.dumps(dataset_info, indent=4), "dataset_info.json")
            
            # Train model
            logging.info("⏳ Training model")
            history = model.fit(
                X_train_scaled, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_val_scaled, y_val),
                verbose=1
            )
            
            # Log metrics
            final_val_loss = history.history['val_loss'][-1]
            final_val_accuracy = history.history['val_accuracy'][-1]
            mlflow.log_metric("val_loss", final_val_loss)
            mlflow.log_metric("val_accuracy", final_val_accuracy)
            
            # Log training history
            history_df = pd.DataFrame(history.history)
            mlflow.log_text(history_df.to_csv(index=False), "training_history.csv")
            
            logging.info("✅ Training completed successfully")
            
            # Save model
            model_path = os.path.join(MODEL_DIR, "loan_approval_model.keras")
            model.save(model_path)
            logging.info(f"✅ Model saved.")
            
            mlflow.tensorflow.log_model(tf_model=model, artifact_path="models", input_example=X_train_scaled.iloc[:1].to_numpy())
            logging.info(f"✅ Model logged.")
            
            logging.info("===================================================================================================")
            return model
    
    except Exception as e:
        logging.error(f"⛔ Error in training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()