# register_model.py

import json
import mlflow
import logging
import dagshub

# ✅ Initialize DagsHub + MLflow
dagshub.init(
    repo_owner='laxmikantbabaleshwar07',
    repo_name='mlops_mini_project',
    mlflow=True
)

# --- Logging configuration ---
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading model info: %s', e)
        raise


def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        # For DagsHub MLflow, just log the model as an artifact
        mlflow.set_experiment("dvc-pipeline")
        with mlflow.start_run(run_id=model_info['run_id']):
            mlflow.log_artifact('model/model.pkl', artifact_path='model')
            logger.info(f" Model artifact logged successfully for run {model_info['run_id']}")
    except Exception as e:
        logger.error('Failed to register model: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
