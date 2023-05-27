"""
Script for reading in data from text files with labels indicated in a given csv file, creates a dataset.DatasetDict
object from the data, and saves it to a given output directory.

Currently expects that the given directory has the following structure:
    ../data/
        [lang]/
            all.csv
            text/
"""

import os
from typing import Any, Optional

from datasets import ClassLabel, DatasetDict, load_dataset, Audio
from transformers import AutoTokenizer, PreTrainedTokenizerBase

# def process_raw_dataset(
#     ds: DatasetDict,
#     class_labels: ClassLabel,
#     tokenizer: Optional[str] = None,
#     drop_file: bool = False,
#     text_file_dir: str = ".",
#     ext: str = "txt",
#     file_encoding: str = "utf-8-sig",
# ) -> DatasetDict:
#     tok = AutoTokenizer(tokenizer) if tokenizer else None
#     def _(
#         entry: dict[str, Any],
#     ) -> dict[str, Any]:
#         fname = f'{entry["file"]}.{ext}'
#         fpath = os.path.join(text_file_dir, fname)
#         with open(fpath, "r", encoding=file_encoding) as f:
#             lines = f.readlines()
#             text = "".join(l.strip() for l in lines)
#             entry["text"] = text
#         entry["label"] = class_labels.str2int(entry["label"].strip().upper())
#         if drop_file:
#             del entry["file"]
#         if tok:
#             entry.update(tok(entry["text"]))
#         return entry
#
#     ds = ds.map(_).cast_column("label", class_labels)
#     # if drop_file:
#     #     ds = ds.remove_columns("file")
#     # if tokenizer:
#     #     ds.map(tokenizer)
#     return ds


def process_raw_dataset(
    entry: dict[str, Any],
    class_labels: ClassLabel,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    drop_file: bool = False,
    text_file_dir: str = ".",
    ext: str = "txt",
    file_encoding: str = "utf-8-sig",
    **tokenizer_kwargs,
) -> dict[str, Any]:
    fname = f'{entry["file"]}.{ext}'
    fpath = os.path.join(text_file_dir, fname)
    with open(fpath, "r", encoding=file_encoding) as f:
        lines = f.readlines()
        text = "".join(l.strip() for l in lines)
        entry["text"] = text
    entry["label"] = class_labels.str2int(entry["label"].strip().upper())
    if drop_file:
        del entry["file"]
    if tokenizer:
        entry.update(tokenizer(entry["text"], **tokenizer_kwargs))
    return entry

    ds = ds.map(_)
    # if drop_file:
    #     ds = ds.remove_columns("file")
    # if tokenizer:
    #     ds.map(tokenizer)
    return ds


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
    class_labels: ClassLabel,
    root_data_dir: str = "../data/",
    subdir: str = "text/",
    data_files_name: str = "all.csv",
    delimiter: str = ",",
    tokenizer: Optional[str] = None,
    **process_raw_dataset_kwargs,
) -> DatasetDict:
    ext = os.path.splitext(data_files_name)[-1][1:]
    lang_dir = os.path.join(root_data_dir, lang)
    text_file_dir = os.path.join(lang_dir, subdir)
    data_file = os.path.join(lang_dir, data_files_name)
    raw_ds = load_dataset(ext, data_files=data_file, delimiter=delimiter)
    audio_file_dir = os.path.join(lang_dir, "audio/")
    labels_file = os.path.join(lang_dir, data_files_name)
    raw_ds = load_dataset(ext, data_files=labels_file, delimiter=delimiter)

    process_raw_dataset_kwargs["text_file_dir"] = text_file_dir
    process_raw_dataset_kwargs["class_labels"] = class_labels
    if tokenizer:
        process_raw_dataset_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(
            tokenizer
        )
    # ds = process_raw_dataset(raw_ds, **process_raw_dataset_kwargs)
    ds = raw_ds.map(
        process_raw_dataset,
        fn_kwargs=process_raw_dataset_kwargs,
    ).cast_column("label", class_labels)

    ds = ds.map(
        add_audio,
        fn_kwargs={"path": audio_file_dir},
    ).cast_column("audio", Audio(sampling_rate=16000))

    return ds
