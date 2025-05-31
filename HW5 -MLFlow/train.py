import os
import psutil
import mlflow
import mlflow.sklearn
import mlflow.catboost
import pandas as pd

from mlflow.tracking import MlflowClient
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score,
    precision_score, recall_score
)

from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

load_dotenv()

username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")


class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [str(x).lower() for x in X]


def run_single_experiment(
        train_df, val_df,
        num_estimates,
        consent_threshold,
        model_type,
        model_params,
        data_processing_params
):
    """
    Запускает один эксперимент, логирует в MLflow
    """

    with mlflow.start_run() as run:
        mlflow.set_tag("experiment", "week5")

        mlflow.log_param("num_estimates", num_estimates)
        mlflow.log_param("consent_threshold", consent_threshold)
        mlflow.log_param("model_type", model_type)

        for k, v in model_params.items():
            mlflow.log_param(f"model_params_{k}", v)

        for k, v in data_processing_params.items():
            mlflow.log_param(f"data_processing_{k}", v)

        if model_type == "logreg":
            model = LogisticRegression(**model_params)
        elif model_type == "catboost":
            model = CatBoostClassifier(**model_params, verbose=0)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        pipeline = Pipeline([
            ("cleaner", TextCleaner()),
            ("tfidf", TfidfVectorizer(**data_processing_params)),
            ("scaler", StandardScaler(with_mean=False)),
            ("model", model),
        ])

        pipeline.fit(train_df["text"], train_df["is_bot"])

        train_preds = pipeline.predict(train_df["text"])
        train_probas = pipeline.predict_proba(train_df["text"])[:, 1]

        train_accuracy = accuracy_score(train_df["is_bot"], train_preds)
        train_logloss = log_loss(train_df["is_bot"], train_probas)
        train_f1 = f1_score(train_df["is_bot"], train_preds)
        train_precision = precision_score(train_df["is_bot"], train_preds)
        train_recall = recall_score(train_df["is_bot"], train_preds)

        val_preds = pipeline.predict(val_df["text"])
        val_probas = pipeline.predict_proba(val_df["text"])[:, 1]

        val_accuracy = accuracy_score(val_df["is_bot"], val_preds)
        val_logloss = log_loss(val_df["is_bot"], val_probas)
        val_f1 = f1_score(val_df["is_bot"], val_preds)
        val_precision = precision_score(val_df["is_bot"], val_preds)
        val_recall = recall_score(val_df["is_bot"], val_preds)

        mlflow.log_metric("train_num_dialogs", len(train_df))
        mlflow.log_metric("val_num_dialogs", len(val_df))

        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("train_logloss", train_logloss)
        mlflow.log_metric("val_logloss", val_logloss)

        mlflow.log_metric("train_f1", train_f1)
        mlflow.log_metric("val_f1", val_f1)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("val_precision", val_precision)
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("val_recall", val_recall)

        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        mlflow.log_metric("cpu_usage", cpu_usage)
        mlflow.log_metric("memory_usage", memory_usage)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        run_id = run.info.run_id
        return run_id, val_logloss


def main():
    tracking_uri = f"http://{username}:{password}@localhost:5050" if username and password else "http://localhost:5050"

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("week5")

    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")

    train_df = train_df.fillna({"text": ""})
    val_df = val_df.fillna({"text": ""})

    experiments_to_run = []
    for ne in [1, 2, 3, 4, 5]:
        for ct in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for mt in ["logreg", "catboost"]:
                if mt == "logreg":
                    model_params = {
                        "C": 1.0,
                        "max_iter": 300
                    }
                else:
                    model_params = {
                        "iterations": 200,
                        "depth": 4
                    }

                data_processing_params = {
                    "ngram_range": (1, 2),
                    "max_df": 0.95
                }

                experiments_to_run.append((
                    ne, ct, mt, model_params, data_processing_params
                ))

    experiments_to_run = experiments_to_run[:10]

    results = []
    for (num_estimates,
         consent_threshold,
         model_type,
         model_params,
         data_processing_params) in experiments_to_run:
        run_id, val_metric = run_single_experiment(
            train_df, val_df,
            num_estimates, consent_threshold,
            model_type, model_params,
            data_processing_params
        )
        results.append((run_id, model_type, val_metric))

    client = MlflowClient(tracking_uri=tracking_uri)
    df_res = pd.DataFrame(results, columns=["run_id", "model_type", "val_logloss"])

    for mt in df_res["model_type"].unique():
        subset = df_res[df_res["model_type"] == mt].sort_values("val_logloss")
        top2 = subset.head(2)
        if len(top2) < 2:
            continue

        champion_run_id = top2.iloc[0]["run_id"]
        challenger_run_id = top2.iloc[1]["run_id"]

        reg_name = f"week5_{mt}"

        try:
            client.get_registered_model(reg_name)
        except Exception:
            client.create_registered_model(reg_name)

        champion_model_uri = f"runs:/{champion_run_id}/model"
        mv_champion = client.create_model_version(
            name=reg_name,
            source=champion_model_uri,
            run_id=champion_run_id
        )
        # Присваиваем alias champion
        client.set_registered_model_alias(
            name=reg_name,
            alias="champion",
            version=mv_champion.version
        )

        challenger_model_uri = f"runs:/{challenger_run_id}/model"
        mv_challenger = client.create_model_version(
            name=reg_name,
            source=challenger_model_uri,
            run_id=challenger_run_id
        )
        # Присваиваем alias challenger
        client.set_registered_model_alias(
            name=reg_name,
            alias="challenger",
            version=mv_challenger.version
        )

        print(f"[INFO] Для '{mt}' назначены champion=run_id({champion_run_id}), challenger=run_id({challenger_run_id})")

    print("[INFO] Эксперименты проведены, лучшие модели зарегистрированы в Model Registry.")
    print("[INFO] Алиасы 'champion' и 'challenger' назначены для каждой пары моделей.")


if __name__ == "__main__":
    main()
