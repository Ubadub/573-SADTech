import os
from typing import Any

import datasets
# from datasets import Features, ClassLabel, Dataset, Value, load_dataset
from sklearn.model_selection import StratifiedKFold
import transformers

CLASS_NAMES = (
    "HIGHLY NEGATIVE",
    "NEGATIVE",
    "NEUTRAL",
    "POSITIVE",
    "HIGHLY POSITIVE",
)

CLASS_LABELS = datasets.ClassLabel(names=CLASS_NAMES)

FEATS = datasets.Features(
    {
        "text": datasets.Value(dtype="string"),
        "label": CLASS_LABELS
    }
)

def process_raw_dataset(entry: dict[str, Any], path=".", ext="txt") -> dict[str, Any]:
    fname = f'{entry["file"]}.{ext}'
    fpath = os.path.join(path, fname)
    with open(fpath, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        text = ''.join(l.strip() for l in lines)
        entry["text"] = text
    entry["label"] = CLASS_LABELS.str2int(entry["label"].strip().upper())
    del entry["file"]
    return entry

def assemble_dataset(
    lang: str,
    root_data_dir: str = "../data/",
    subdir: str = "text/",
    labels_file_name: str = "all.csv",
    delimiter: str = ",",
) -> datasets.Dataset:
    ext = os.path.splitext(labels_file_name)[-1][1:]
    lang_dir = os.path.join(root_data_dir, lang)
    text_file_dir = os.path.join(lang_dir, subdir)
    labels_file = os.path.join(lang_dir, labels_file_name)
    raw_ds = datasets.load_dataset(ext, data_files=labels_file, delimiter=delimiter)
#    raw_ds = datasets.load_dataset("csv", data_files="../data/tam/all.csv", delimiter=",")
#    raw_ds = datasets.load_dataset(labels_file.split(".")[-1], )

    ds = raw_ds.map(process_raw_dataset, fn_kwargs={"path" : text_file_dir}).cast_column("label", CLASS_LABELS)
    return ds
