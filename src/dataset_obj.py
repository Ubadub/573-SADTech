"""
Hugging Face Dataset Object Creator
Reads in .txt files, partitions them based on their train/dev/test split,
and creates a Hugging Face datasets object for use in downstream tasks
"""
import os
import csv
import pandas as pd
from datasets import Features, ClassLabel, Dataset, Value
from sklearn.model_selection import train_test_split


def _get_txt_file_list(dir_path: str) -> list[tuple[str, str]]:
    """
    :param dir_path: Takes a path to a directory containing all language data
    :return: A list of all file names in the text/ directory for the given language
        mapped to their file contents
    """
    txt_dir_path = os.path.join(dir_path, "text")

    file_names = []
    file_contents = []
    for file in os.listdir(txt_dir_path):
        if not file.endswith(".txt"):
            continue
        txt_file_path = os.path.join(txt_dir_path, file)
        file_name = file.split(".")[0]
        file_names.append(file_name)

        with open(txt_file_path) as f:
            text = f.read()
            file_contents.append(text.strip())

    file_repr = zip(file_names, file_contents)
    file_repr = sorted(file_repr, key=lambda item: item[0])
    return file_repr


def get_data(dir_path: str, lang: str = "") -> pd.DataFrame:
    """
    :param dir_path: Takes a path to a directory containing all language data
    :param lang: Takes a language string (e.g. "tam" or "mal")
    :return: A pandas DataFrame mapping mapping file names to their respective text and labels
    """
    dir_path = os.path.join(dir_path, lang)
    file_repr = _get_txt_file_list(dir_path)
    files_df = pd.DataFrame(file_repr, columns=["file_name", "text"])

    gold_labels_file_path = os.path.join(dir_path, "labels.tsv")
    gold_labels = []
    with open(gold_labels_file_path) as gold_labels_file:
        gold_labels_csv = csv.reader(gold_labels_file, delimiter="\t")
        header = True
        for line in gold_labels_csv:
            if header:
                header = False
                continue
            line[1] = line[1].upper()
            line = tuple(line[:2])
            gold_labels.append(line)

    gold_labels_df = pd.DataFrame(gold_labels, columns=["file_name", "label"])
    dataset_df = gold_labels_df.merge(files_df, on="file_name")

    return dataset_df


def create_dataset_obj(dir_path: str, lang: str = "") -> tuple[Dataset, Dataset]:
    """
    :param dir_path: Takes a path to a directory containing all language data
    :param lang: Takes a language string (e.g. "tam" or "mal")
    :return: A tuple containing Train and Dev Hugging Face dataset objects for use in downstream tasks
    """
    dataset_df = get_data(dir_path, lang)

    # perform train dev split on data
    x_train_df, x_dev_df, y_train_df, y_dev_df =\
        train_test_split(dataset_df[["file_name", "text"]],
                         dataset_df[["file_name", "label"]], test_size=0.2, random_state=2020)

    train_df = y_train_df.merge(x_train_df, on="file_name")
    dev_df = y_dev_df.merge(x_dev_df, on="file_name")

    # string labels for sentiment, mapping 0 --> HIGHLY NEGATIVE, 1 --> NEGATIVE, etc.
    class_names = ["HIGHLY NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "HIGHLY POSITIVE"]
    features = Features({'file_name': Value(dtype='string'),
                           'label': ClassLabel(num_classes=5, names=class_names), 'text': Value(dtype='string')})

    # create Dataset object for train and dev
    train_dataset = Dataset.from_pandas(train_df, split="train", features=features)
    dev_dataset = Dataset.from_pandas(dev_df, split="dev", features=features)

    return train_dataset, dev_dataset



if __name__ == "__main__":
    pass
    # for testing
    # test_dir = "data"
    # lang = "mal"
    #
    # # get_data(test_dir, lang)
    # create_dataset_obj(test_dir, lang)
