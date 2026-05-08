import mlflow

# Define the experiment name
mlflow.set_experiment("Initial_Setup_Check")

# Start a 'run' (like a single page in your diary)
with mlflow.start_run():
    mlflow.log_param("setup_step", "Phase_1")
    mlflow.log_metric("environment_ready", 1.0)
    print("Successfully logged to MLflow!")