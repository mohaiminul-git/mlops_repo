import json

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import yaml
from src.logger import logging
import dagshub
import mlflow
import os



# dagshub.init(
#     repo_owner="mohaiminul-git",
#     repo_name="mlops_repo",
#     mlflow=True
# )
# dagsub_token = os.getenv("DAGSUB_TOKEN") 
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSUB_TOKEN")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSUB_TOKEN")

dagshub_url = "https://dagshub.com"
repo_owner = "mohaiminul-git"
repo_name = "mlops_repo"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
        clf.fit(X_train, y_train)
        logging.info('Model training completed')
        return clf
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise
    
def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise
def main():
    mlflow.set_experiment("my-dvc-pipeline_1")
    
    # Keep the try-except completely outside the run context to catch tracking initialization issues too
    try:
        train_data = load_data('./data/processed/train_bow.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)
        
        with mlflow.start_run() as run:  
            # Force log_model to execute cleanly. If this fails, the script will crash here and show why.
            mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model"
                #registered_model_name="my_model"
            )
                        
            save_model(clf, 'models/model.pkl')
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
    except Exception as e:
        logging.error('CRITICAL: Process failed: %s', e)
        raise e # <--- Force the exception to bubble up so you can see it in the terminal!

if __name__ == '__main__':
    main()