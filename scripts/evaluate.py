import mlflow
import mlflow.tensorflow
import tensorflow as tf
import pandas as pd
import logging
import os

from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix
from utils import load_preprocessed_data
from config import LOG_DIR, MLFLOW_DB_PATH, TEST_PATH

# Setup directories and logging
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

def evaluate_and_test(model_name="LoanApprovalModel", version=None, alias=None):
    """Evaluate a registered model and generate predictions on test data."""
    try:
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        mlflow.set_experiment("Loan_Prediction")

        # Load preprocessed data
        logging.info("⏳ Loading validation and test data")
        _, X_val_scaled, _, y_val, test_data_scaled = load_preprocessed_data()

        # Load model
        client = mlflow.tracking.MlflowClient()
        if alias:
            logging.info(f"⏳ Loading model '{model_name}' with alias '{alias}'")
            version_info = client.get_model_version_by_alias(model_name, alias)
            version = version_info.version
            model_uri = f"models:/{model_name}@{alias}"
        else:
            if not version:
                version = client.get_latest_versions(model_name)[0].version
                logging.info(f"Using latest model version: {version}")
            model_uri = f"models:/{model_name}/{version}"

        try:
            model = mlflow.tensorflow.load_model(model_uri)
            logging.info("✅ Model Loaded")
        except Exception as e:
            logging.error(f"⛔ Error during model loading: {str(e)}")
            raise

        with mlflow.start_run(run_name="evaluation_and_testing"):
            # Evaluate on validation data
            logging.info("⏳ Evaluating model on validation data")
            loss, accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
            y_pred_proba = model.predict(X_val_scaled, verbose=0).flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)

            auc = roc_auc_score(y_val, y_pred_proba)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)

            # Log evaluation metrics
            mlflow.log_metric("val_loss", loss)
            mlflow.log_metric("val_accuracy", accuracy)
            mlflow.log_metric("val_auc", auc)
            mlflow.log_metric("val_precision", precision)
            mlflow.log_metric("val_recall", recall)
            mlflow.log_param("evaluation_batch_size", 32)
            mlflow.log_param("classification_threshold", 0.5)

            logging.info(f"✅ Evaluation: Loss={loss:.4f}, Accuracy={accuracy:.4f}, AUC={auc:.4f}, "
                         f"Precision={precision:.4f}, Recall={recall:.4f}")

            # Log confusion matrix
            cm = confusion_matrix(y_val, y_pred)
            logging.info(f"🌐 Confusion Matrix: {cm}")
            
            # Predict on test data
            logging.info("⏳ Generating predictions on test data")
            predictions_proba = model.predict(test_data_scaled, verbose=0).flatten()
            predictions = (predictions_proba > 0.5).astype(int)

            pred_df = pd.DataFrame({
                "probability": predictions_proba,
                "prediction": predictions,
                "label": ["Yes" if p == 1 else "No" for p in predictions]
            })

            mlflow.log_text(
                pd.concat([pd.read_csv(TEST_PATH, index_col='id').reset_index(), pred_df], axis=1).to_csv(index=False),
                "test_predictions.csv"
            )

            # Log tags
            mlflow.set_tag("run_type", "evaluation_and_testing")
            mlflow.set_tag("model_name", model_name)
            mlflow.set_tag("model_version", version)
            if alias:
                mlflow.set_tag("model_alias", alias)

            logging.info("✅ All artifacts and metrics logged to MLflow")
            logging.info("===================================================================================================")

            return {
                "evaluation_metrics": {
                    "loss": loss,
                    "accuracy": accuracy,
                    "auc": auc,
                    "precision": precision,
                    "recall": recall
                },
                "test_predictions": pred_df
            }

    except Exception as e:
        logging.error(f"⛔ Error during evaluation or testing: {str(e)}")
        raise

if __name__ == "__main__":
    evaluate_and_test(model_name="LoanApprovalModel", alias="ReadyForProduction")