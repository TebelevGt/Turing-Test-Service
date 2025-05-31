import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from params import Params

data_dir = "/home/georgii-tebelev/g.tebelev/data/raw_data"
all_data = []


# Recursive
for root, _dirs, files in os.walk(data_dir):
    for file in files:
        file_path = Path(root) / file  # Use Path with `/` operator
        with file_path.open(encoding="utf-8") as f:  # Use Path.open() instead of open()
            data = json.load(f)
            all_data.append(data)


# Create dataframe
data = pd.DataFrame(columns=["text", "label"])
for i in range(len(all_data)):
    if all_data[i]["result"][0]["from_name"] == "participant_1_label":
        if all_data[i]["result"][0]["value"]["choices"] == ["Bot"]:
            p1 = 1
            p0 = 0
        else:
            p1 = 0
            p0 = 1
    elif all_data[i]["result"][0]["value"]["choices"] == ["Bot"]:
        p1 = 0
        p0 = 1
    else:
        p1 = 1
        p0 = 0
    for j in range(len(all_data[i]["task"]["data"]["messages"])):
        texts = all_data[i]["task"]["data"]["messages"]

        if texts[j]["participant_index"] == "0":
            data.loc[len(data)] = {"text": texts[j]["timed_text"].split("]")[1], "label": p0}
        else:
            data.loc[len(data)] = {"text": texts[j]["timed_text"].split("]")[1], "label": p1}

# Split data within the MLflow run
train_idx, test_idx = next(
    StratifiedShuffleSplit(
        n_splits=1,
        test_size=Params.TEST_SIZE,
        random_state=Params.RANDOM_STATE,
    ).split(data, data["label"].to_list()),
)

# CSV SAVE
train_df = data.iloc[train_idx.tolist()]
test_df = data.iloc[test_idx.tolist()]

folder_name = "/home/georgii-tebelev/g.tebelev/data/preprocessed_data"

# Check if folder exist
folder_path = Path(folder_name)
if not folder_path.exists():
    folder_path.mkdir(parents=True, exist_ok=True)

train_df.to_csv("/home/georgii-tebelev/g.tebelev/data/preprocessed_data/train.csv")
test_df.to_csv("/home/georgii-tebelev/g.tebelev/data/preprocessed_data/test.csv")
