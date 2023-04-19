"""
Classifier
"""

import sys
import re
import yaml

import datasets

import vectors

from spacy.lang.ta import Tamil
from spacy.lang.ml import Malayalam

TAM_STOP_WORDS = Tamil().Defaults.stop_words
MAL_STOP_WORDS = Malayalam().Defaults.stop_words

def clean_up_line(line: str, config: dict) -> str:
    """
    Given a string and a set of stop words, removes extraneous punctuation,
    numbers, and the stop words from the string.

    Returns the cleaned up string.
    """
    line = line + " "

    if config["remove_punc"]:
        line = re.sub(r",|;|\"|\'|\?|:|!", "", line)    # removes punctuation
        line = re.sub(r"\\", "", line)                  # removes punctuation
        line = re.sub(r"\.\s", " ", line)               # removes punctuation

    if config["remove_stop_words"]:
        stop_words = TAM_STOP_WORDS if config["lang"] == "tam" else MAL_STOP_WORDS
        line = " ".join([word for word in line.split()
                            if word not in stop_words]) # removes stop words

    if config["remove_num"]:
        line = re.sub(r"\s\S*?\d\S*?\s", " ", line)     # removes words that contain numbers
        line = re.sub(r"\d", "", line)                  # removes numbers

    # line = re.sub(r"[a-zA-Z]", " ", line)             # removes single letters

    return line.strip()


def do_preprocessing(config: dict, ds_dict: datasets.DatasetDict) -> None:
    pass

def write_output(config):
    with open(config["output_path"], "w") as wf:
        pass


def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(config["data_path"])

    if config["remove_punc"] or config["remove_stop_words"] or config["remove_num"]:
        do_preprocessing(config, ds_dict)

    get_vectors() # based on which model it is; use vectors.py

    if config["model"] == "base":
        do_baseline()
        # build up a class

    do_inference()

    write_output()

    # model = config["model"]


if __name__ == '__main__':
    main()