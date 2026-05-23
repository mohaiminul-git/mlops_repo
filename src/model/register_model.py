import json
import os
import pickle
import logging
import warnings
import mlflow
from mlflow import MlflowClient
from src.logger import logging

# Ignore unnecessary warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Production setup for DagsHub/MLflow
# -------------------------------------------------------------------------------------
dagshub_token = os.getenv("DAGSUB_TOKEN")
if not dagshub_token:
    raise EnvironmentError("DAGSUB_TOKEN environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "mohaiminul-git"
repo_name = "mlops_repo"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry and transition it to Staging."""
    try:
        # Construct the unique MLflow run URI for the model artifact
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        # Register the model artifact
        logging.info(f"Registering model from URI: {model_uri}")
        model_version = mlflow.register_model(model_uri, model_name)

        # Extract the new version string safely
        latest_version = model_version.version

        # Initialize MLflow client to handle stage transition
        client = MlflowClient()

        # Transition the newly registered version to "Staging"
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Staging"
        )
        logging.info(f'Model "{model_name}" version {latest_version} successfully registered and transitioned to Staging.')
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

        
# import json
# import os
# import mlflow
# from mlflow import MlflowClient
# import logging
# from src.logger import logging
# import dagshub
# import warnings

# warnings.simplefilter("ignore", UserWarning)
# warnings.filterwarnings("ignore")

# # dagshub.init(
# #     repo_owner="mohaiminul-git",
# #     repo_name="mlops_repo",
# #     mlflow=True
# # )

# os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSUB_TOKEN")
# os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSUB_TOKEN")

# dagshub_url = "https://dagshub.com"
# repo_owner = "mohaiminul-git"
# repo_name = "mlops_repo"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

# def main():
#     try:
#         model_name = "my_model"
#         # Define the exact version you want to target
#         target_version = "4"  

#         client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
        
#         # Verify the version exists before moving it
#         model_version_details = client.get_model_version(name=model_name, version=target_version)
#         #print(model_version_details)
#         logging.info(f"Found version {model_version_details.version} (Status: {model_version_details.status})")

#         # Promote this specific version to Staging
#         client.set_registered_model_alias(
#             name=model_name,
#             alias="production",
#             version=str(target_version)
#         )
#         logging.info(f"Successfully transitioned '{model_name}' version {target_version} to Production.")

#     except Exception as e:
#         logging.error('Failed to complete the model registration process: %s', e)
# if __name__ == '__main__':
#     main()

