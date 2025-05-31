import json
from pathlib import Path

import pandas as pd
import pytest
from pandas import DataFrame

from turing_test_service.evaluate import get_predictions_and_metrics, pipeline


# Мокируем данные для тестирования
@pytest.fixture
def mock_dataset() -> None:
    data = {"text": ["sample text 1", "sample text 2", "sample text 3"], "label": [0, 1, 0]}
    return pd.DataFrame(data)


# Тестируем функцию get_predictions_and_metrics
def test_get_predictions_and_metrics(mock_dataset: DataFrame) -> None:
    metrics = get_predictions_and_metrics(pipeline, mock_dataset)

    # Проверяем, что метрики возвращаются и содержат ожидаемые ключи
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1-score" in metrics
    assert "support" in metrics


def test_metrics_file_creation() -> None:
    metrics_file = Path("/home/georgii-tebelev/g.tebelev/metrics/metrics.json")
    assert metrics_file.exists()

    # Проверяем, что файл содержит корректные данные
    with metrics_file.open("r") as json_file:
        metrics = json.load(json_file)
        assert "train_precision" in metrics
        assert "test_precision" in metrics
