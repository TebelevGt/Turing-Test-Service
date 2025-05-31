import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from turing_test_service.dataset import download_all_objects


@pytest.fixture
def mock_s3() -> None:
    with patch("turing_test_service.dataset.boto3.client") as mock_client:
        # Создаем мок для s3
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        yield mock_s3


def test_download_all_objects(mock_s3: MagicMock) -> None:
    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()

    try:
        # Настройка моков
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        # Имитируем возвращаемые данные от S3
        mock_paginator.paginate.return_value = [{"Contents": [{"Key": "file1.txt"}, {"Key": "file2.txt"}]}]

        # Вызов тестируемой функции
        download_all_objects("test-bucket", "", temp_dir)

        # Проверка вызовов
        mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
        mock_s3.download_file.assert_any_call("test-bucket", "file1.txt", f"{temp_dir}/file1.txt")
        mock_s3.download_file.assert_any_call("test-bucket", "file2.txt", f"{temp_dir}/file2.txt")

    finally:
        # Удаляем временную директорию после завершения теста
        shutil.rmtree(temp_dir)


def test_download_all_objects_no_files(mock_s3: MagicMock) -> None:
    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()

    try:
        # Настройка моков
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        # Имитируем пустой результат от S3
        mock_paginator.paginate.return_value = []

        # Вызов тестируемой функции
        download_all_objects("test-bucket", "", temp_dir)

        # Проверка вызовов
        mock_s3.get_paginator.assert_called_once_with("list_objects_v2")
        mock_s3.download_file.assert_not_called()

    finally:
        # Удаляем временную директорию после завершения теста
        shutil.rmtree(temp_dir)


def test_download_all_objects_create_directories(mock_s3: MagicMock) -> None:
    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp()

    try:
        # Настройка моков
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator

        # Имитируем возвращаемые данные от S3
        mock_paginator.paginate.return_value = [{"Contents": [{"Key": "subdir/file1.txt"}]}]

        # Вызов тестируемой функции
        download_all_objects("test-bucket", "", temp_dir)

        # Проверка вызовов
        mock_s3.download_file.assert_called_once_with("test-bucket", "subdir/file1.txt", f"{temp_dir}/subdir/file1.txt")
        assert Path(f"{temp_dir}/subdir").exists()

    finally:
        # Удаляем временную директорию после завершения теста
        shutil.rmtree(temp_dir)
