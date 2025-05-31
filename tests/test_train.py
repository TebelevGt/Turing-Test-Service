from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Импортируем функции из train.py
from turing_test_service.train import folder_name, pipeline, train_df


def test_folder_creation() -> None:
    """Проверяем, что папка для моделей создается, если она не существует."""
    folder_path = Path(folder_name)
    assert folder_path.exists(), "Папка для моделей не была создана"


def test_data_loading() -> None:
    """Проверяем, что данные загружаются корректно."""
    assert not train_df.empty, "Данные не были загружены или DataFrame пуст"


def test_pipeline_creation() -> None:
    """Проверяем, что pipeline создается корректно."""
    assert isinstance(pipeline, Pipeline), "Pipeline не был создан"
    assert isinstance(pipeline.steps[0][1], TfidfVectorizer), "TfidfVectorizer не был добавлен в pipeline"
    assert isinstance(pipeline.steps[1][1], LogisticRegression), "LogisticRegression не был добавлен в pipeline"


def test_model_training() -> None:
    """Проверяем, что модель обучается корректно."""
    x_train = train_df["text"].to_list()
    y_train = train_df["label"].to_list()
    pipeline.fit(X=x_train, y=y_train)
    assert hasattr(pipeline, "classes_"), "Модель не была обучена"


def test_model_saving() -> None:
    """Проверяем, что модель сохраняется корректно."""
    model_path = Path(folder_name) / "trained_model.joblib"
    joblib.dump(pipeline, model_path)
    assert model_path.exists()
