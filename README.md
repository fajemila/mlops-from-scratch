# MLOps From Scratch: AQI Prediction Pipeline

## 🚀 Project Overview
This project is an end-to-end Machine Learning Operations (MLOps) pipeline designed to predict Air Quality Index (PM2.5) using weather sensor data (temperature, wind speed, pressure, etc.). It demonstrates industry-standard practices for tracking experiments, versioning data, orchestrating pipelines, ensuring code quality, and automating testing and builds.

## 🛠️ Tech Stack
* **Environment Management:** `uv`
* **Data & Pipeline Versioning:** `DVC` (Data Version Control)
* **Experiment Tracking:** `MLflow`
* **Model Serving:** `FastAPI` & `Uvicorn`
* **Containerization:** `Docker`
* **CI/CD:** GitHub Actions
* **Code Quality & Testing:** `Ruff`, `pytest`

## 📁 Project Structure

```text
mlops-from-scratch/
├── .github/workflows/
│   └── ci.yml             # GitHub Actions CI/CD pipeline
├── data/                  # DVC-tracked data directory (raw & processed)
├── notebooks/             # Initial EDA and baseline modeling
├── src/
│   ├── api/               # FastAPI application for model serving
│   ├── data/              # Data preprocessing scripts
│   └── models/            # Model training scripts
├── tests/                 # Unit tests using pytest
├── .dvc/                  # DVC configuration
├── mlruns/                # MLflow local tracking directory
├── Dockerfile             # Containerization blueprint
├── dvc.yaml               # DVC pipeline orchestration definitions
├── params.yaml            # Hyperparameter configuration
├── pyproject.toml         # Python dependencies managed by uv
└── README.md