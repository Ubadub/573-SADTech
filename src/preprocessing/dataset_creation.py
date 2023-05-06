"""
Script for reading in data from text files with labels indicated in a given csv file, creates a dataset.DatasetDict
object from the data, and saves it to a given output directory.

Currently expects that the given directory has the following structure:
    ../data/
        [lang]/
            all.csv
            text/
"""

import argparse
import os
from typing import Any

import datasets

from common import CLASS_LABELS, GLOBAL_SEED, N_FOLDS


def process_raw_dataset(
    entry: dict[str, Any],
    class_labels: datasets.ClassLabel,
    path: str = ".",
    ext: str = "txt",
    file_encoding: str = "utf-8-sig",
) -> dict[str, Any]:
    fname = f'{entry["file"]}.{ext}'
    fpath = os.path.join(path, fname)
    with open(fpath, "r", encoding=file_encoding) as f:
        lines = f.readlines()
        text = "".join(l.strip() for l in lines)
        entry["text"] = text
    entry["label"] = class_labels.str2int(entry["label"].strip().upper())
    # del entry["file"]
    return entry


def add_audio(
    entry: dict[str, Any],
    path: str = ".",
    ext: str = "mp3",
) -> dict[str, Any]:
    fname = f'{entry["file"]}.{ext}'
    fpath = os.path.join(path, fname)
    entry["audio"] = fpath
    return entry


def assemble_dataset(
    lang: str,
    class_labels: datasets.ClassLabel,
    root_data_dir: str = "../data/",
    subdir: str = "text/",
    labels_file_name: str = "all.csv",
    delimiter: str = ",",
) -> datasets.DatasetDict:
    ext = os.path.splitext(labels_file_name)[-1][1:]
    lang_dir = os.path.join(root_data_dir, lang)
    text_file_dir = os.path.join(lang_dir, subdir)
    audio_file_dir = os.path.join(lang_dir, "audio/")
    labels_file = os.path.join(lang_dir, labels_file_name)
    raw_ds = datasets.load_dataset(ext, data_files=labels_file, delimiter=delimiter)

    ds = raw_ds.map(
        process_raw_dataset,
        fn_kwargs={"path": text_file_dir, "class_labels": class_labels},
    ).cast_column("label", class_labels)

    ds = ds.map(
        add_audio,
        fn_kwargs={"path": audio_file_dir},
    ).cast_column("audio", datasets.Audio(sampling_rate=16000))

    return ds


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Generate a `datasets.DatasetDict` from the text files in a given language's data directory and save it to file.",
    )
    parser.add_argument("lang")
    parser.add_argument("-o", "--output")
    #    parser.add_argument("-d", "--data_dir", default="../data/")
    #    parser.add_argument("-l", "--labels_file", default="all.csv")
    #    parser.add_argument("-m", "--delimiter", default=",")

    args = parser.parse_args()
    lang = args.lang
    output_path = args.output or f"../data/{lang}/train_dataset_dict"
    output_dir_path = os.path.dirname(output_path)

    if output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)

    ds_dict: datasets.DatasetDict = assemble_dataset(lang, CLASS_LABELS)
    ds_dict.save_to_disk(output_path)


if __name__ == "__main__":
    main()
