import os
import mlflow

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    # MLflow tracking setup
    dagshub_url = "https://dagshub.com"
    repo_owner = "laxmikantbabaleshwar07"
    repo_name = "mlops_mini_project"
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = mlflow.MlflowClient()
    model_name = "my_model"

    # Safely get the latest version in Staging
    try:
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    except Exception as e:
        print(f"Error fetching staging versions: {e}")
        return

    if not staging_versions:
        print(f"No models in Staging to promote for {model_name}. Exiting.")
        return

    latest_version_staging = staging_versions[0].version

    # Archive current Production models
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version.version,
                stage="Archived"
            )
        except Exception as e:
            print(f"Warning: Could not archive version {version.version}: {e}")

    # Promote Staging model to Production
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
        )
        print(f"Model version {latest_version_staging} promoted to Production")
    except Exception as e:
        print(f"Error promoting model version {latest_version_staging}: {e}")

if __name__ == "__main__":
    promote_model()
