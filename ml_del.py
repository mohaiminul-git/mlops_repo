import mlflow
from mlflow import MlflowClient
import os

os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSUB_TOKEN")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSUB_TOKEN")

repo_owner = "mohaiminul-git"
repo_name = "mlops_repo"

# IMPORTANT
mlflow.set_tracking_uri(
    f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
)

client = MlflowClient()

# List models
models = client.search_registered_models()

print("Found models:", len(models))

# Delete models
for model in models:
    name = model.name
    print("Deleting:", name)

    client.delete_registered_model(name)

print("All models deleted.")