# fetch_artifacts.py
import mlflow
import os
import shutil
from mlflow.tracking import MlflowClient

def fetch_artifacts():
    client = MlflowClient()

    # Define experiment name and metric
    experiment_name = "Liver Disease Prediction"
    metric_name = "roc_auc"
    ascending = False

    # Get the experiment ID from the experiment name
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

    # Search for the best run based on the specified metric
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )

    if runs:
        best_run = runs[0]
        
        # Create Model directory if it does not exist
        os.makedirs("Model", exist_ok=True)

        # Download and copy the model
        model_path = mlflow.artifacts.download_artifacts(run_id=best_run.info.run_id, artifact_path="random_forest_model")
        print( model_path           )
        shutil.copy(f'{model_path}/model.pkl', "Model/rf_model.pkl")

        # Download and copy the scaler
        scaler_path = mlflow.artifacts.download_artifacts(run_id=best_run.info.run_id, artifact_path="scaler.pkl")
        shutil.copy(scaler_path, "Model/scaler.pkl")

        print("Model and scaler downloaded and copied to 'Model' directory.")
    else:
        print("No runs found.")

if __name__ == "__main__":
    fetch_artifacts()