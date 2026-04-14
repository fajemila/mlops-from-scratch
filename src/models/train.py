import pandas as pd
import numpy as np
import mlflow
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train_and_log_model(data_dir: str, n_estimators: int, max_depth: int):
    # Load the processed data
    train = pd.read_csv(f"{data_dir}/train.csv", index_col="date")
    test = pd.read_csv(f"{data_dir}/test.csv", index_col="date")

    X_train, y_train = train.drop(columns=["pm2.5"]), train["pm2.5"]
    X_test, y_test = test.drop(columns=["pm2.5"]), test["pm2.5"]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("AQI_Production_Pipeline")

    with mlflow.start_run(run_name="RandomForest_Script"):
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})

        model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        print(f"✅ Model trained! RMSE: {rmse:.2f}, MAE: {mae:.2f}")
        mlflow.log_metrics({"rmse": rmse, "mae": mae})
        mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    # Read parameters from the YAML file
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    # Extract the specific variables
    n_est = params["train"]["n_estimators"]
    depth = params["train"]["max_depth"]

    # Pass them into the function
    train_and_log_model("data/processed", n_estimators=n_est, max_depth=depth)
