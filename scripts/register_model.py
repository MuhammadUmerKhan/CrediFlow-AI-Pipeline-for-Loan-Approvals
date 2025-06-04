import mlflow
import mlflow.tensorflow
import logging
import os
import argparse
import mlflow.tracking
from config import LOG_DIR, MLFLOW_DB_PATH

os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "app.log")),
        logging.StreamHandler()
    ]
)

def register_model(run_id, model_name, alias, description, experiment_name):
    """Register a model to MLflow Model Registry with aliases and description."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
        mlflow.set_experiment(experiment_name)
        
        # Get the latest run if run_id is not provided
        if not run_id:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            runs = client.search_runs(experiment_ids=[experiment.experiment_id], max_results=1, order_by=["start_time DESC"])
            if not runs:
                raise ValueError("No runs found in the Loan_Prediction experiment")
            run_id = runs[0].info.run_id
            logging.info(f"Using latest run ID: {run_id}")
        else:
            logging.info(f"Using provided run ID: {run_id}")
        
        # Register the model
        logging.info(f"⏳ Registering model '{model_name}' from run {run_id}")
        model_uri = f"runs:/{run_id}/models"
        registered_model = mlflow.register_model(model_uri, model_name)
        
        # Set aliases
        client = mlflow.tracking.MlflowClient()
        if alias:
            client.set_registered_model_alias(
                name=model_name,
                alias=alias,
                version=registered_model.version
            )
            logging.info(f"✅ Set alias '{alias}' for model version {registered_model.version}")
        
        # Set description
        if description:
            client.update_model_version(
                name=model_name,
                version=registered_model.version,
                description=description
            )
            logging.info(f"✅ Set description for model version {registered_model.version}")
        
        # Add tags
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="dataset",
            value="Loan Dataset"
        )
        client.set_model_version_tag(
            name=model_name,
            version=registered_model.version,
            key="model_type",
            value="TensorFlow"
        )
        
        logging.info(f"✅ Model '{model_name}' registered as version {registered_model.version}")
        logging.info("===================================================================================================")
        return registered_model
    
    except Exception as e:
        logging.error(f"⛔ Error registering model: {str(e)}")
        raise

if __name__=="__main__":
    description = "The LoanApprovalModel is a TensorFlow neural network designed for binary classification to predict loan approval outcomes ('Yes' or 'No') based on applicant data."
    alias = "ReadyForProduction"
    name = "LoanApprovalModel"
    experiment_name = "Loan_Prediction"
    register_model(run_id=None, model_name=name, alias=alias, description=description, experiment_name=experiment_name)