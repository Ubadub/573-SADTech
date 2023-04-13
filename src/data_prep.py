"""
Data Preparation
Tokenizes the data and divides it, as well as the labels, into test/validation
splits. Vectorizes the data and outputs the vectors as sparse matrices into
.npz files. Also creates and outputs a master .csv file that houses all the data
to the data/directory
"""

import os
import re
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import spmatrix

from spacy.lang.ta import Tamil
from spacy.lang.ml import Malayalam


def clean_up_line(line: str, stop_words: set) -> str:
    """
    Given a string and a set of stop words, removes extraneous punctuation,
    numberes, and the stop words from the string.

    Things removed:
        punctuation: . , ; " ' ? : !
        numbers: 0-9

    Returns the cleaned up string.
    """
    line = line + " "
    line = re.sub(r"\s\S*?\d\S*?\s", " ", line)
    # line = re.sub(r"[a-zA-Z]", " ", line)
    line = re.sub(r",|;|\"|\'|\?|\d|:|!", "", line)
    line = re.sub(r"\\", "", line)
    line = re.sub(r"\.\s", " ", line)
    line = " ".join([word for word in line.split()
                        if word not in stop_words])
    return line.strip()


def get_file_words(directory: str, stop_words: set) -> dict[str, str]:
    """
    Given a directory path and a set of stop words, cleans up each text
    document in the directory.

    Returns a dictionary mapping each document's name to a string of the words
    in said document.
    """
    file_words = {}
    for file in os.listdir(directory):
        file_path = directory + file
        file_name = file.split(".")[0]
        file_words[file_name] = ""
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                line = clean_up_line(line, stop_words)
                file_words[file_name] += " " + line
            file_words[file_name] = file_words[file_name].strip()
    return file_words


def vectorize(X_train: spmatrix, X_val: spmatrix) -> tuple[spmatrix, spmatrix]:
    """
    Given train and validation splits, creates tfidf vectors of each split (from
    https://skimai.com/fine-tuning-bert-for-sentiment-analysis/).

    Returns the vectors as sparse matrices.
    """
    # Preprocess text
    X_train_preprocessed = np.array(X_train)
    X_val_preprocessed = np.array(X_val)

    # Calculate TF-IDF
    tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                            binary=True,
                            smooth_idf=False)
    X_train_tfidf = tf_idf.fit_transform(X_train_preprocessed)
    X_val_tfidf = tf_idf.transform(X_val_preprocessed)

    return X_train_tfidf, X_val_tfidf


def get_merged_df(lang: str, stop_words: set) -> pd.DataFrame:
    """
    Given a language string and a set of stop words of the given language, creates a
    Pandas DataFrame object housing ["File name", "Lang", "Label", "Text"] columns for all
    the data of the given language

    Returns the cleaned data as a Pandas DataFrame object
    """
    values = get_file_words("data/" + lang + "/text/", stop_words)
    values_temp = {"File name": list(values.keys()), "Text": list(values.values())}
    values_df = pd.DataFrame.from_dict(values_temp)

    labels_df = pd.read_csv("data/" + lang + "/" + lang + "_train_label.tsv", sep="\t", header=0)

    merged = labels_df.merge(values_df, left_on="File name", right_on="File name")
    merged["Lang"] = lang
    merged = merged[["File name", "Lang", "Label", "Text"]]

    return merged


def get_vectors(lang: str, merged: pd.DataFrame) -> None:
    """
    Given a language string and Pandas DataFrame representing all the data of the given langauge,
    creates train/validation splits for them, and vectorizes the values.

    Saves sparse matrices as ".npz" files containing the vectorized data for the test and
    validation splits to the outputs/vectors/ directory
    """
    X_train, X_val, y_train, y_val =\
    train_test_split(merged["Text"], merged["Label"], test_size=0.2, random_state=2020)

    X_train_tfidf, X_val_tfidf = vectorize(X_train, X_val)

    np.savez("outputs/vectors/" + lang + "_X_train_tfidf.npz", data=X_train_tfidf.data, indices=X_train_tfidf.indices,
             indptr=X_train_tfidf.indptr, shape=X_train_tfidf.shape)
    np.savez("outputs/vectors/" + lang + "_X_val_tfidf.npz", data=X_val_tfidf.data, indices=X_val_tfidf.indices,
             indptr=X_val_tfidf.indptr, shape=X_val_tfidf.shape)


def main():
    TAM_STOP_WORDS = Tamil().Defaults.stop_words
    tam_merged = get_merged_df("tam", TAM_STOP_WORDS)
    # tam_merged.to_csv("data/tam/tam_data_with_labels.csv", header=False, index=False)
    get_vectors("tam", tam_merged)

    MAL_STOP_WORDS = Malayalam().Defaults.stop_words
    mal_merged = get_merged_df("mal", MAL_STOP_WORDS)
    # mal_merged.to_csv("data/mal/mal_data_with_labels.csv", header=False, index=False)
    get_vectors("mal", mal_merged)

    master_merged = pd.concat([tam_merged, mal_merged])
    master_merged.to_csv("data/master_data_with_labels.csv", header=False, index=False)


if __name__ == '__main__':
    main()
