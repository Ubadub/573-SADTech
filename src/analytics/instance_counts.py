import sys
import datasets
import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Optional

AXIS_FONT_SIZE = 16


def create_instance_count_plot(
        ds_dict: datasets.arrow_dataset.Dataset,
        lang: str,
        stage: str,
        output_path: Optional[str] = "analytics"
) -> None:
    if lang == "mal":
        lang = "Malayalam"
    else:
        lang = "Tamil"

    classes: list = ds_dict.features["label"].names

    train_df: pd.DataFrame = ds_dict.to_pandas()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.countplot(
        x=train_df["label"],
        palette='viridis'
    )

    ax.set_xticklabels(classes)
    ax.bar_label(ax.containers[0])
    plt.xticks(rotation=45)

    plt.xlabel("Class", fontsize=AXIS_FONT_SIZE)
    plt.ylabel("Class Count", fontsize=AXIS_FONT_SIZE)
    title = f"{lang} {stage} Data Class Frequency"
    plt.title(title, fontsize=AXIS_FONT_SIZE)



    output_path = os.path.join(output_path, title)
    plt.savefig(output_path, bbox_inches="tight")


if __name__ == '__main__':
    lang = sys.argv[1]
    ds_dict_path = f"../data/{lang}/train_dataset_dict"
    ds_dict: datasets.DatasetDict = datasets.load_from_disk(ds_dict_path)

    ds_dict_train = ds_dict["train"]
    ds_dict_test = ds_dict["test"]

    try:
        output_path = sys.argv[2]
        create_instance_count_plot(ds_dict_train, lang, "Train", output_path)
        create_instance_count_plot(ds_dict_test, lang, "Test", output_path)
    except:
        create_instance_count_plot(ds_dict_train, lang, "Train")
        create_instance_count_plot(ds_dict_test, lang, "Test")
