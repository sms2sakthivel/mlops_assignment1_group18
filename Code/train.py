import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE 
from sklearn.ensemble import RandomForestClassifier
from loguru import logger
from pickle import dump
import mlflow
import mlflow.sklearn

logger.add("./logs.log", level="TRACE", format="{time} {message}", retention="2500000 seconds")
test_ratio = 0.2
random_state=100

def load_dataset():
    df = pd.read_csv("./Data/liver_disease.csv")
    logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
    return df

def preprocess_data(df):
    # Step 1: Handle missing values
    df["Albumin_and_Globulin_Ratio"].ffill(inplace=True)

    # Step 2: Label encoding
    X = df.iloc[:,:-1]
    Y = df.iloc[:,[-1]]
    Y = Y["Dataset"].map({"Yes":1, "No":0}) # Yes means patient has liver disease.

    # Step 3: Splitting the dataset
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y , test_size=test_ratio , random_state=random_state)

    # Step 4: Fixing class imbalance using SMOTE
    # The dataset is highly imbalanced, There are 416 patients with liver disease and 167 patients without liver disease.
    sm = SMOTE(random_state = 25, sampling_strategy = 0.75)
    X_train, Y_train = sm.fit_resample(X_train, Y_train)

    # Step 5: Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Scale the test set as well

    # Step 6: Creating final training dataset
    X_train_final = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    logger.info(f"Data Preprocessing completed successfully.")
    return X_train_final, Y_train, X_test_final, Y_test, scaler

def train_model(X_train, Y_train):
    # Step 1: Define Random Forest Classifier model
    rf_model = RandomForestClassifier()

    # Step 2: Define hyperparameters to tune
    param_grid = {
        "n_estimators":[200,250,300],
        "max_depth":[3,6,9,12],
        "min_samples_leaf":[5,10,15,20]
    }

    # Step 3: Define GridSearchCV Hyperparameter tuning object
    rf_grid = GridSearchCV(rf_model, param_grid = param_grid, cv=5, refit=True, scoring = 'roc_auc', verbose=3)

    # Step 4: Train the model
    rf_grid_fit = rf_grid.fit(X_train,Y_train)

    logger.info(f"Model Training completed successfully.")
    return rf_grid_fit


def evaluate_model(rf_grid_fit, X_test, Y_test):
    # Step 1: Make predictions
    Y_pred = rf_grid_fit.predict(X_test)
    Y_pred_proba = rf_grid_fit.predict_proba(X_test)[:, 1]

    # Step 2: Evaluate the model
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred_proba)
    cm = confusion_matrix(Y_test, Y_pred)

    # Step 3: Print or log the results
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    logger.info(f"Model Evaluation completed successfully.")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }

def save_model(rf_grid_fit, scaler):
    dump(scaler, open('./Model/scaler.pkl', 'wb'))
    dump(rf_grid_fit, open('./Model/rf_model.pkl', 'wb'))
    logger.info(f"Model and Scaler saved successfully.")



def train():
    # Set the experiment
    mlflow.set_experiment("Liver Disease Prediction")
    
    # Start an MLflow run
    with mlflow.start_run():
        # Set tags
        mlflow.set_tag("project", "liver_disease")
        mlflow.set_tag("author", "MLOPS-Group-18")
        
        # Step 1: Load the dataset
        df = load_dataset()

        # Step 2: Preprocess the data
        X_train, Y_train, X_test, Y_test, scaler = preprocess_data(df)

        # Step 3: Tune Hyper Parameters and Train the model
        rf_grid_fit = train_model(X_train, Y_train)

        # Step 4: Evaluate the model
        evaluation_results = evaluate_model(rf_grid_fit, X_test, Y_test)

        # Step 5: Save the model and scaler
        save_model(rf_grid_fit, scaler)

        # Step 6: Log the parameters, metrics, and artifacts to MLflow
        mlflow.log_param("Training Dataset Size", len(X_train))
        mlflow.log_param("Test Dataset Size", len(X_test))
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_ratio", test_ratio)
        mlflow.log_param("best_n_estimators", rf_grid_fit.best_params_["n_estimators"])
        mlflow.log_param("best_max_depth", rf_grid_fit.best_params_["max_depth"])
        mlflow.log_param("best_min_samples_leaf", rf_grid_fit.best_params_["min_samples_leaf"])
        mlflow.log_metric("best_score", rf_grid_fit.best_score_)  # Log the best score
        mlflow.log_metric("accuracy", evaluation_results["accuracy"])
        mlflow.log_metric("precision", evaluation_results["precision"])
        mlflow.log_metric("recall", evaluation_results["recall"])
        mlflow.log_metric("f1_score", evaluation_results["f1_score"])
        mlflow.log_metric("roc_auc", evaluation_results["roc_auc"])

        # Log confusion matrix as a CSV artifact
        cm_df = pd.DataFrame(evaluation_results["confusion_matrix"], index=["Actual Yes", "Actual No"], columns=["Pred Yes", "Pred No"])
        cm_df.to_csv("confusion_matrix.csv")
        mlflow.log_artifact("confusion_matrix.csv")

        # Log the trained model
        mlflow.sklearn.log_model(rf_grid_fit, "random_forest_model")

        # Log the scaler
        mlflow.log_artifact("./Model/scaler.pkl")

        # Step 7: Log the results in the logger
        logger.info(f"Best Hyper Parameters : {rf_grid_fit.best_params_} with Score : {rf_grid_fit.best_score_}")
        logger.info(f"Cross Validation Detailed Results : {rf_grid_fit.cv_results_}")
        logger.info(f"Model Evaluation Metrics : {evaluation_results}")

if __name__ == "__main__":
    train()