import mlflow

# This will create a 'mlruns' folder locally to store your logs
mlflow.set_experiment("Loan_Default_Prediction")

with mlflow.start_run():
    mlflow.log_param("data_version", "raw_v1")
    mlflow.log_metric("status", 1)
    print("MLflow tracking is initialized and working!")