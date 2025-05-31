from pathlib import Path

import pandas as pd

from turing_test_service.train_test_split import all_data, data, test_df, train_df


def test_data_loading() -> None:
    """Тест загрузки данных из JSON-файлов."""
    assert len(all_data) > 0, "Данные не были загружены"


def test_dataframe_creation() -> None:
    """Тест создания DataFrame."""
    assert isinstance(data, pd.DataFrame), "Данные не были преобразованы в DataFrame"
    assert "text" in data.columns, "Столбец 'text' отсутствует в DataFrame"
    assert "label" in data.columns, "Столбец 'label' отсутствует в DataFrame"


def test_train_test_split() -> None:
    """Тест разделения данных на тренировочную и тестовую выборки."""
    assert len(train_df) > 0, "Тренировочная выборка пуста"
    assert len(test_df) > 0, "Тестовая выборка пуста"
    assert len(train_df) + len(test_df) == len(data), "Разделение данных выполнено некорректно"


def test_csv_saving() -> None:
    """Тест сохранения данных в CSV-файлы."""
    train_csv_path = Path("/home/georgii-tebelev/g.tebelev/data/preprocessed_data/train.csv")
    test_csv_path = Path("/home/georgii-tebelev/g.tebelev/data/preprocessed_data/test.csv")
    assert train_csv_path.exists(), "Тренировочный CSV-файл не был создан"
    assert test_csv_path.exists(), "Тестовый CSV-файл не был создан"

    train_csv_data = pd.read_csv(train_csv_path)
    test_csv_data = pd.read_csv(test_csv_path)
    assert len(train_csv_data) == len(train_df), "Данные в тренировочном CSV не соответствуют ожидаемым"
    assert len(test_csv_data) == len(test_df), "Данные в тестовом CSV не соответствуют ожидаемым"
