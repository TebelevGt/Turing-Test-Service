import os
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

username = os.getenv("MLFLOW_TRACKING_USERNAME")
password = os.getenv("MLFLOW_TRACKING_PASSWORD")
print(username, password)
tracking_uri = f"http://{username}:{password}@localhost:5050"


def main():
    mlflow.set_tracking_uri(tracking_uri)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dialog_path = os.path.join(base_dir, "sample_dialog.json")

    model_name = "week5_logreg@champion"
    model_uri = f"models:/{model_name}"

    model = mlflow.sklearn.load_model(model_uri)

    with open(dialog_path, "r", encoding="utf-8") as f:
        dialog_data = json.load(f)

    messages = dialog_data["task"]["data"]["messages"]

    if messages and "created_at" in messages[0]:
        messages.sort(key=lambda msg: datetime.strptime(msg["created_at"], "%Y-%m-%d %H:%M:%S.%f"))

    for msg in messages:
        text = msg.get("text", "")
        participant_index = msg.get("participant_index", "")
        prob_is_bot = model.predict_proba([text])[0, 1]

        print(f"{participant_index}: {text}")
        print(f"is_bot_probability: {prob_is_bot}")


if __name__ == "__main__":
    main()