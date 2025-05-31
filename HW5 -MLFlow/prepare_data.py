import os
import json
import argparse
import pandas as pd

from collections import defaultdict, Counter


def load_annotations(annotation_dir: str):
    """
    Рекурсивно читает все файлы из annotation_dir.
    Возвращает список всех аннотаций.
    """
    all_data = []
    for root, dirs, files in os.walk(annotation_dir):
        for filename in files:
            if filename.endswith(""):
                path = os.path.join(root, filename)
                with open(path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                    all_data.append(ann)
    return all_data


def gather_by_dialog_id(annotations_list):
    """
    Группирует аннотации по dialog_id
    Возвращает dict: {dialog_id: [annotation1, annotation2, ...], ...}
    """
    grouped = defaultdict(list)
    for ann in annotations_list:
        dialog_data = ann.get("task", {}).get("data", {})
        dialog_id = dialog_data.get("dialog_id", None)
        if dialog_id is not None:
            grouped[dialog_id].append(ann)
    return grouped


def compute_consensus(annotations_for_dialog):
    """
    Для данного диалога собираем голоса:
      - participant_0_label (Human/Bot)
      - participant_1_label (Human/Bot)
    Возвращаем:
      votes_p0 = список ["Human" или "Bot"] для участника 0
      votes_p1 = список ["Human" или "Bot"] для участника 1
    """
    votes_p0 = []
    votes_p1 = []
    for ann in annotations_for_dialog:
        result_list = ann.get("result", [])
        for r in result_list:
            from_name = r.get("from_name", "")
            choices = r.get("value", {}).get("choices", [])
            if not choices:
                continue
            label = choices[0]
            if from_name == "participant_0_label":
                votes_p0.append(label)
            elif from_name == "participant_1_label":
                votes_p1.append(label)
    return votes_p0, votes_p1


def get_majority_label(votes_list):
    """
    Принимает список строк ["Bot","Human","Bot",...] 
    """
    if not votes_list:
        return None, 0
    counter = Counter(votes_list)
    label, freq = counter.most_common(1)[0]
    ratio = freq / len(votes_list)
    return label, ratio


def build_dataset(annotations_list, num_estimates, consent_threshold):
    """
    Собирает общий train/val DataFrame. 
    """
    grouped = gather_by_dialog_id(annotations_list)

    train_records = []
    val_records = []

    for dialog_id, ann_list in grouped.items():
        if len(ann_list) < num_estimates:
            selected_label0 = "Human"
            selected_label1 = "Bot"

            messages = ann_list[0]["task"]["data"]["messages"]
            for msg in messages:
                p_idx = msg.get("participant_index", "0")
                text = msg.get("text", "")
                if p_idx == "0":
                    is_bot = 1 if selected_label0 == "Bot" else 0
                else:
                    is_bot = 1 if selected_label1 == "Bot" else 0
                train_records.append((dialog_id, text, is_bot))
            continue

        votes_p0, votes_p1 = compute_consensus(ann_list)
        label0, ratio0 = get_majority_label(votes_p0)
        label1, ratio1 = get_majority_label(votes_p1)

        is_consensus = (ratio0 >= consent_threshold) and (ratio1 >= consent_threshold)

        if not label0:
            label0 = "Human"
        if not label1:
            label1 = "Human"

        if is_consensus:
            target_list = val_records
        else:
            target_list = train_records

        messages = ann_list[0]["task"]["data"]["messages"]
        for msg in messages:
            p_idx = msg.get("participant_index", "0")
            text = msg.get("text", "")
            if p_idx == "0":
                is_bot = 1 if label0 == "Bot" else 0
            else:
                is_bot = 1 if label1 == "Bot" else 0
            target_list.append((dialog_id, text, is_bot))

    df_train = pd.DataFrame(train_records, columns=["dialog_id", "text", "is_bot"])
    df_val = pd.DataFrame(val_records, columns=["dialog_id", "text", "is_bot"])
    return df_train, df_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotation_dir", type=str, default="annotations",
                        help="Путь к папке с размеченными файлами")
    parser.add_argument("--num_estimates", type=int, default=2,
                        help="Мин. кол-во аннотаций для диалога")
    parser.add_argument("--consent_threshold", type=float, default=0.8,
                        help="Доля голосов для определения (Bot/Human).")
    parser.add_argument("--train_out", type=str, default="data/train.csv",
                        help="Куда сохранить train.csv")
    parser.add_argument("--val_out", type=str, default="data/val.csv",
                        help="Куда сохранить val.csv")
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)

    all_anns = load_annotations(args.annotation_dir)
    df_train, df_val = build_dataset(all_anns, args.num_estimates, args.consent_threshold)
    df_train.to_csv(args.train_out, index=False)
    df_val.to_csv(args.val_out, index=False)

    print(f"[INFO] Train size = {len(df_train)}, Val size = {len(df_val)}")
    print("[INFO] Готово! Датасеты сохранены:")