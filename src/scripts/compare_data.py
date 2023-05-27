"""
"""

import os
import pandas as pd


def get_df(directory):
    files = os.listdir(directory)
    data_dict = {"file_name": [], "text_data": []}
    for file in files:
        fname = file.split(".")[0]
        fpath = os.path.join(directory, file)
        with open(fpath, encoding="utf-8-sig") as f:
            lines = f.readlines()
            text = "".join(l.strip() for l in lines)
            data_dict["file_name"].append(fname)
            data_dict["text_data"].append(text)
    df = pd.DataFrame(data_dict)
    return df


def get_merged_df(lang):
    real_labels = pd.read_csv(f"../data/{lang}/all.csv")
    real_labels = real_labels.rename(
        columns={"file": "file_name", "label": "label_real"}
    )
    real_df = get_df(f"../data/{lang}/text")
    real_df = real_df.merge(real_labels, on="file_name")

    test_df = get_df(f"../data/test_data/{lang}/text")
    test_merged_df = real_df.merge(
        test_df, on="text_data", how="outer", suffixes=["_real", "_test"]
    )

    train_labels = None
    if lang == "mal":
        xlsx = "MAL_MSA_labels.xlsx"
        train_labels = pd.read_excel(f"../data/train_data/{lang}/{xlsx}")
        train_labels = train_labels.rename(
            columns={"File name": "file_name", "Labels": "label_train"}
        )
    elif lang == "tam":
        xlsx = "TAM_MSA_label.xlsx"
        train_labels = pd.read_excel(f"../data/train_data/{lang}/{xlsx}")
        train_labels = train_labels.rename(
            columns={"File name": "file_name", "labels": "label_train"}
        )

    train_df = get_df(f"../data/train_data/{lang}/text")
    train_df = train_df.merge(train_labels, on="file_name")
    train_merged_df = real_df.merge(
        train_df, on="text_data", how="outer", suffixes=["_real", "_train"]
    )

    merged_df = train_merged_df.merge(
        test_merged_df, on=["text_data", "file_name_real", "label_real"], how="outer"
    )
    merged_df = merged_df[
        [
            "file_name_real",
            "label_real",
            "file_name_train",
            "label_train",
            "file_name_test",
        ]
    ]
    merged_df = merged_df.sort_values(
        ["file_name_real", "file_name_train", "file_name_test"]
    )

    return merged_df


def main():
    mal_merged_df = get_merged_df("mal")
    mal_merged_df.to_csv("../data/mal/merged.csv", index=False)
    with open("../data/mal/merged.txt", "w") as wf:
        wf.write(mal_merged_df.to_string(index=False))

    tam_merged_df = get_merged_df("tam")
    tam_merged_df.to_csv("../data/tam/merged.csv", index=False)
    with open("../data/tam/merged.txt", "w") as wf:
        wf.write(tam_merged_df.to_string(index=False))


if __name__ == "__main__":
    main()
