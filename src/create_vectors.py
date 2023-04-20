"""
Vectors
"""

import sys
import re
import yaml

import datasets
import numpy as np

from spacy.lang.ta import Tamil
from spacy.lang.ml import Malayalam
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix


TAM_STOP_WORDS = Tamil().Defaults.stop_words
MAL_STOP_WORDS = Malayalam().Defaults.stop_words

def clean_up_text(line: str, config: dict) -> str:
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


def do_preprocessing(ds_dict: datasets.DatasetDict, config: dict) -> datasets.DatasetDict:
    """
    Param ds_dict: Takes in a huggingface object
    Param config: Takes in a config dictionary

    Performs preprocessing on all text data. Cleans text data by removing stopwords
    and/or punctuation based on values from the config dictionary.

    Returns a datasetDict with cleaned text
    """
    if config["remove_punc"] or config["remove_stop_words"] or config["remove_num"]:
        ds_dict["text"] = clean_up_text(ds_dict["text"], config)
    return ds_dict


def tfidf_vectors(ds_dict: datasets.DatasetDict):
    text = np.array(ds_dict["train"]["text"])
    tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                             binary=True,
                             smooth_idf=False
                             )
    tfidf_vectors = tf_idf.fit_transform(text)

    return tfidf_vectors


def create_vectors(ds_dict: datasets.DatasetDict, config: dict) -> spmatrix:
    if config["vectors"] == "tfidf":
        # Calculate TF-IDF Vectors
        vectors = tfidf_vectors(ds_dict)

    # otherwise not a known vectorization method
    else:
        raise ValueError(f"config argument {config['vectors']} not a known vectorization method")

    return vectors



def main():
    config_file = sys.argv[1]
    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(config["data_path"])

    ds_dict["train"] = ds_dict["train"].map(do_preprocessing,
                                            fn_kwargs={"config": config}
                                            )

    vectors = create_vectors(ds_dict, config)

    print(vectors)
    # we want to turn data into vectors

    # model = config["model"]


if __name__ == '__main__':
    main()