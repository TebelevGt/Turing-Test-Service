from pathlib import Path

import boto3
from botocore.client import Config

# Yandex Object Storage
endpoint_url = "https://storage.yandexcloud.net"
bucket_name = "dialog-annotations"
folder_name = "/home/georgii-tebelev/g.tebelev/data/raw_data"


# S3 credentials (добавьте свои реальные ключи)
aws_access_key_id = "YCAJEVR2fGI4PIxHLurCaHUEE"
aws_secret_access_key = "YCMJSxtUIjFmDrOlyFBuR-HuYpO4EtbaTh1cFEXQ"

# Check if folder exists
folder_path = Path(folder_name)
if not folder_path.exists():
    folder_path.mkdir(parents=True, exist_ok=True)

# S3 init with credentials
session = boto3.session.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)
s3 = session.client(
    service_name="s3",
    endpoint_url=endpoint_url,
    config=Config(signature_version="s3v4")
)

def download_all_objects(bucket: str, prefix: str, local_dir: str) -> None:
    try:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    local_file_path = Path(local_dir) / key
                    local_file_path.parent.mkdir(parents=True, exist_ok=True)
                    s3.download_file(bucket, key, str(local_file_path))
        print("Download completed successfully")
    except Exception as e:
        print(f"Error during download: {str(e)}")
        raise

# Скачиваем все объекты из бакета
download_all_objects(bucket_name, "", folder_name)
