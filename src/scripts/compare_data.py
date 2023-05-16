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
    real_labels = real_labels.rename(columns={"file": "file_name", "label": "label_real"})
    real_df = get_df(f"../data/{lang}/text")
    real_df = real_df.merge(real_labels, on="file_name")

    test_df = get_df(f"../data/test_data/{lang}/text")
    test_merged_df = real_df.merge(test_df, on="text_data", how="outer", suffixes=["_real", "_test"])
    test_merged_df = test_merged_df[["file_name_real", "label_real", "file_name_test"]]
    test_merged_df = test_merged_df.sort_values(["file_name_real", "file_name_test"])

    train_labels = None
    if lang == "mal":
        xlsx = "MAL_MSA_labels.xlsx"
        train_labels = pd.read_excel(f"../data/train_data/{lang}/{xlsx}")
        train_labels = train_labels.rename(columns={"File name": "file_name", "Labels": "label_train"})
    elif lang == "tam":
        xlsx = "TAM_MSA_label.xlsx"
        train_labels = pd.read_excel(f"../data/train_data/{lang}/{xlsx}")
        train_labels = train_labels.rename(columns={"File name": "file_name", "labels": "label_train"})

    train_df = get_df(f"../data/train_data/{lang}/text")
    train_df = train_df.merge(train_labels, on="file_name")
    train_merged_df = real_df.merge(train_df, on="text_data", how="outer", suffixes=["_real", "_train"])
    train_merged_df = train_merged_df[["file_name_real", "label_real", "file_name_train", "label_train"]]
    train_merged_df = train_merged_df.sort_values(["file_name_real", "file_name_train"])

    return (test_merged_df, train_merged_df)


def main():
    # (mal_test_merged_df, mal_train_merged_df) = get_merged_df("mal")
    # mal_test_merged_df.to_csv("../data/mal/test_merged.csv", index=False)
    # mal_train_merged_df.to_csv("../data/mal/train_merged.csv", index=False)

    # (tam_test_merged_df, tam_train_merged_df) = get_merged_df("tam")
    # tam_test_merged_df.to_csv("../data/tam/test_merged.csv", index=False)
    # tam_train_merged_df.to_csv("../data/tam/train_merged.csv", index=False)

    print("##### Mal Train Data Merged #####")
    print(pd.read_csv("../data/mal/train_merged.csv").to_string())
    print()
    print("##### Tam Train Data Merged #####")
    print(pd.read_csv("../data/tam/train_merged.csv").to_string())


if __name__ == "__main__":
    main()