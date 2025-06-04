import logging
import os
from preprocess import preprocess_data
from train import train_model
from register_model import register_model
from evaluate import evaluate_and_test
from config import LOG_DIR

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

def run_pipeline(model_name="LoanApprovalModel", alias="ReadyForProduction", description=None, save_model=False):
    """Run the entire ML pipeline: preprocess, train, register, evaluate, and test."""
    try:
        logging.info("🚀 Starting ML pipeline")
        
        # Step 1: Preprocess data
        logging.info("⏳ Running data preprocessing")
        preprocess_data()
        logging.info("✅ Preprocessing completed")
        
        # Step 2: Train model
        logging.info("⏳ Training model")
        train_model()
        logging.info("✅ Model trained")
        
        # Step 3: Register model
        logging.info("⏳ Registering model")
        default_description = "The LoanApprovalModel is a TensorFlow neural network designed for binary classification to predict loan approval outcomes ('Yes' or 'No') based on applicant data."
        logging.warning("⚠️ Using latest MLflow run for registration, as run_id is not provided. Ensure no concurrent runs.")
        registered_model = register_model(
            run_id=None,
            model_name=model_name,
            alias=alias,
            description=default_description,
            experiment_name="Loan_Prediction"
        )
        logging.info(f"✅ Model registered as {model_name} version {registered_model.version}")
        
        # Step 4: Evaluate and test model
        logging.info("⏳ Evaluating and testing model")
        results = evaluate_and_test(
            model_name=model_name,
            alias=alias
        )
        logging.info("✅ Evaluation and testing completed")
        logging.info(f"Evaluation metrics: {results['evaluation_metrics']}")
        
        logging.info("🎉 Pipeline completed successfully")
        logging.info("===================================================================================================")
        
        return {
            "model_name": model_name,
            "model_version": registered_model.version,
            "evaluation_metrics": results["evaluation_metrics"],
            "test_predictions": results["test_predictions"]
        }
    
    except Exception as e:
        logging.error(f"⛔ Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    
    run_pipeline()