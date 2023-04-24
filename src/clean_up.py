"""
Line Clean Up
"""

import sys
import re
import yaml

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


def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)


if __name__ == '__main__':
    main()