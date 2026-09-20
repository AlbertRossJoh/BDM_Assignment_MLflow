# Running the project

This document covers everything needed to reproduce training, register the resulting model, serve it, and query the prediction endpoint.

## Prerequisites

- Python 3.13 (declared in `python_env.yaml`)
- MLflow 3.15.2 and the remaining pinned dependencies from `requirements.txt`

To install into the current environment:

```bash
pip install -r requirements.txt
```

The input data is committed under `data/`:

| File | Contents |
| --- | --- |
| `data/power.csv` | Measured power generation, sampled every minute |
| `data/weather.csv` | Wind speed and direction forecasts, sampled every three hours |
| `data/future.csv` | Unseen future weather forecasts used to check the served model |

## 1. Start the tracking server

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --host 127.0.0.1 --port 5000
```

The UI is then available at <http://127.0.0.1:5000>.

## 2. Train

Run it from a clean checkout with:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow run . -e main \
  --experiment-name run-mlproject \
  --run-name "SVR train test"
```

Both input paths can be overridden:

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow run . -e main \
  -P power_csv=data/power.csv \
  -P weather_csv=data/weather.csv
```

`train.py` can also be called directly, bypassing MLflow Projects, with the same options:

```bash
python train.py --power-csv data/power.csv --weather-csv data/weather.csv
```

## 3. Serve

```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow models serve \
  -m "models:/<experiment-name>/<version>" \
  -p 5001 --env-manager local
```

