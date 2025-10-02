import dagshub
import mlflow
dagshub.init(repo_owner='laxmikantbabaleshwar07', repo_name='mlops_mini_project', mlflow=True)

mlflowset_tracking_uri = dagshub.get_tracking_uri('https://dagshub.com/laxmikantbabaleshwar07/mlops_mini_project.mlflow')
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)