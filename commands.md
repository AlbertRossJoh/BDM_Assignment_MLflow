
Serve SVR power output prediction version 2:
```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow models serve -m "models:/SVR power output prediction/1" -p 5001 --env-manager local
```

Run train.py
```bash
MLFLOW_TRACKING_URI=http://127.0.0.1:5000 mlflow run . -e main --env-manager local --experiment-name run-mlproject --run-name "SVR train test"
```
