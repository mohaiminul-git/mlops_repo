# import mlflow
# from mlflow import MlflowClient
# import dagshub

# # 1. Initialize DagsHub first to set up the remote tracking URI & credentials
# dagshub.init(
#     repo_owner="mohaiminul-git",
#     repo_name="mlops_repo",
#     mlflow=True
# )

# # 2. Instantiate the client AFTER DagsHub has configured the environment
# client = MlflowClient()

# # 3. Now it will successfully look for the run on DagsHub
# print(client.list_artifacts("adb0782b2e5549559043b6d15a1b7da7"))


import mlflow
import dagshub

dagshub.init(repo_owner="mohaiminul-git", repo_name="mlops_repo", mlflow=True)
client = mlflow.MlflowClient(tracking_uri=mlflow.get_tracking_uri())

# List everything at the root of your run
artifacts = client.list_artifacts("6ef8873b26934920a331b7e7d1fc3ee4")
print("Root artifacts:")
for art in artifacts:
    print(f"- {art.path} (Directory: {art.is_dir})")