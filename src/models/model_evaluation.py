import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import dagshub
import os

# -------------------- DagsHub MLflow Setup --------------------
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = "laxmikantbabaleshwar07"  # ✅ username
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com/laxmikantbabaleshwar07/mlops_mini_project.mlflow"
mlflow.set_tracking_uri(dagshub_url)  # ✅ Fixed

# -------------------- Logging --------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_evaluation_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------------- Helper Functions --------------------
def load_model(file_path: str):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    logger.debug('Model loaded from %s', file_path)
    return model

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    logger.debug('Data loaded from %s', file_path)
    return df

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    metrics_dict = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_pred_proba)
    }
    logger.debug('Model evaluation metrics calculated')
    return metrics_dict

def save_metrics(metrics: dict, file_path: str) -> None:
    with open(file_path, 'w') as file:
        json.dump(metrics, file, indent=4)
    logger.debug('Metrics saved to %s', file_path)

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    model_info = {'run_id': run_id, 'model_path': model_path}
    with open(file_path, 'w') as file:
        json.dump(model_info, file, indent=4)
    logger.debug('Model info saved to %s', file_path)

# -------------------- Main --------------------
def main():
    mlflow.set_experiment("dvc-pipeline")

    with mlflow.start_run() as run:
        clf = load_model('./model/model.pkl')
        test_data = load_data('./data/processed/test_bow.csv')

        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        metrics = evaluate_model(clf, X_test, y_test)
        save_metrics(metrics, 'reports/metrics.json')

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        if hasattr(clf, 'get_params'):
            for k, v in clf.get_params().items():
                mlflow.log_param(k, v)

        save_model_info(run.info.run_id, "model", 'reports/model_info.json')

        mlflow.log_artifact('reports/metrics.json')
        mlflow.log_artifact('reports/model_info.json')
        mlflow.log_artifact('model_evaluation_errors.log')

        logger.info("✅ Model evaluation completed and logged successfully.")

if __name__ == '__main__':
    main()
