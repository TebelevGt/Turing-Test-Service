import os
import secrets
import time
import uuid

import pandas as pd
import requests

SERVICE_HOSTNAME = os.environ["SERVICE_HOSTNAME"]
SERVICE_PORT = os.environ["SERVICE_PORT"]
SENTENCES_PATH = os.environ["SHOOTER_SENTENCES_PATH"]

sentences_df = pd.read_csv(SENTENCES_PATH, sep="\t", header=None)


def get_random_sentence() -> str:
    return sentences_df.sample(1).iloc[0, 0]


def make_request() -> requests.Response:
    data = {
        "text": get_random_sentence(),
        "dialog_id": str(uuid.uuid4()),
        "id": str(uuid.uuid4()),
        "participant_index": secrets.randbelow(3),
    }
    return requests.post(
        f"http://{SERVICE_HOSTNAME}:{SERVICE_PORT}/predict",
        headers={
            "Content-Type": "application/json",
        },
        json=data,
        timeout=5,
    )


def main() -> None:
    while True:
        time.sleep(2)


if __name__ == "__main__":
    main()
