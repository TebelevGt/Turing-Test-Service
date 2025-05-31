#!/usr/bin/env python3

# Standard Library Imports
import os
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold

# Third-Party Imports
import mlflow
import mlflow.sklearn
import joblib
import shutil
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv

# Local Imports
from params import Params

folder_name = "/home/georgii-tebelev/g.tebelev/models"
TOKEN_PATTERN = os.getenv("TOKEN_PATTERN", r"(\w{1,4})\w*\b")

# Check if folder exists
folder_path = Path(folder_name)
if not folder_path.exists():
    folder_path.mkdir(parents=True, exist_ok=True)

# Read data
train_df = pd.read_csv("/home/georgii-tebelev/g.tebelev/data/preprocessed_data/train.csv")
X_train = train_df["text"].to_list()
y_train = train_df["label"].to_list()


load_dotenv()
username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")

tracking_uri = f"http://{username}:{password}@localhost:5050"


# Download model
mlflow.set_tracking_uri(tracking_uri)

base_dir = os.path.dirname(os.path.abspath(__file__))
dialog_path = os.path.join(base_dir, "sample_dialog.json")

model_name = "week5_logreg@champion"
model_uri = f"models:/{model_name}"

best_model = mlflow.sklearn.load_model(model_uri)

# Train best model on full data
best_model.fit(X_train, y_train)

# Сохраняем модель через MLflow
model_path = "/home/georgii-tebelev/g.tebelev/inference_service/models"
if os.path.exists(model_path):
    shutil.rmtree(model_path)
mlflow.sklearn.save_model(best_model, model_path)

print("Model saved successfully")
