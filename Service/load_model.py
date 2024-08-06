import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from pickle import load

experiment_name = "Liver Disease Prediction"
metric_name = "accuracy"


def get_best_run(experiment_name, metric_name, ascending=False):
    # Step 1: Initialize the MLflow client
    client = MlflowClient()
    
    # Step 2: Get the experiment ID from the experiment name
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    
    # Step 3: Search for the best run based on the specified metric
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
        max_results=1
    )
    
    if runs:
        best_run = runs[0]
        return best_run
    else:
        return None

def load_best_model_and_scaler(is_local_model=True):
    global scaler, model
    if is_local_model:
        # Step 1: Load the model and scaler from local files
        model_path = "./Model/rf_model.pkl"
        scaler_path = "./Model/scaler.pkl"

        with open(scaler_path, 'rb') as f:
            scaler = load(f)
        
        with open(model_path, 'rb') as f:
            model = load(f)
    else:
        # Step 1: Query and Retrieve the best run
        best_run = get_best_run(experiment_name=experiment_name, metric_name=metric_name)
        if best_run is None:
            raise Exception("No best run found.")

        # Step 2: Load the best model
        model_uri = f"runs:/{best_run.info.run_id}/random_forest_model"
        model = mlflow.sklearn.load_model(model_uri)

        # Step 3: Load the scaler
        local_path = mlflow.artifacts.download_artifacts(run_id=best_run.info.run_id, artifact_path='scaler.pkl')
        scaler = load(open(local_path, 'rb'))
    return model, scaler
