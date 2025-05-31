import json
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

# Load from file
pipeline =  mlflow.sklearn.load_model("/home/georgii-tebelev/g.tebelev/inference_service/models/")

# Load test collection
train_df = pd.read_csv("/home/georgii-tebelev/g.tebelev/data/preprocessed_data/train.csv")
test_df = pd.read_csv("/home/georgii-tebelev/g.tebelev/data/preprocessed_data/test.csv")


# Calculate metrics
def get_predictions_and_metrics(pipeline: Pipeline, dataset: DataFrame) -> dict:
    predictions = pipeline.predict(dataset["text"].to_list())
    return classification_report(
        y_true=dataset["label"].to_list(),
        y_pred=predictions,
        output_dict=True,
    )["macro avg"]


train_metrics = get_predictions_and_metrics(pipeline, train_df)
test_metrics = get_predictions_and_metrics(pipeline, test_df)

metrics = {f"train_{k}": v for k, v in train_metrics.items()} | {f"test_{k}": v for k, v in test_metrics.items()}


folder_name = "metrics"
# Check if exist
folder_path = Path(folder_name)
if not folder_path.exists():
    folder_path.mkdir(parents=True, exist_ok=True)

# Save json
file_path = Path("/home/georgii-tebelev/g.tebelev/load-train-eval-champ-model/metrics/metrics.json")
with file_path.open("w") as json_file:
    json.dump(metrics, json_file, indent=4)
