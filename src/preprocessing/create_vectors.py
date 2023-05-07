"""
Vectors

A wrapper class for feature vectors. Takes in data and creates vectors based on the specified
    vectorization method

We initialize and obtain document Vectors object as follows:

    vector_wrapper = Vectors(config=config, ds_dict=ds_dict, vec_type="tfidf")
    feature_vectors = vector_wrapper.get_vectors()

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
from typing import Optional, Union


class Vectors:
    TAM_STOP_WORDS = Tamil.Defaults.stop_words
    MAL_STOP_WORDS = Malayalam.Defaults.stop_words

    def __init__(
        self,
        config: Union[dict, str],
        ds_dict: Optional[Union[datasets.DatasetDict, str]] = None,
        vec_type: Optional[str] = None,
    ) -> None:
        """
        Params:
            - config:
                A python dictionary representing the specified config.yml file
                    or
                A file path to a specified config.yml file
            - ds_dict:
                A datasets.DatasetDict object of text data to be vectorized
                    or
                A file path to a saved datasets.DatasetDict of text data to be vectorized
                    or
                If not passed a datasets.DatasetDict object, looks in the config file for a file path
                    to a saved datasets.DatasetDict object
            - vec_type:
                A string specifying which type of vectorization should occur
                    or
                If not passed a string, looks in the config file for a specified type of vectorization

        Initializes the vector wrapper class
        """
        self.config = Vectors._open_config(config)
        self.ds_dict = self._open_ds_dict(ds_dict)
        self.vec_type = self.vec_type(vec_type)
        self.vectors = self.create_vectors()

    @staticmethod
    def _open_config(config: Union[dict, str]) -> dict:
        """
        Params:
            - config: Takes either a config.yml file path as a string or a config file representation as a
                dictionary

        If passed a config file path, will read its contents

        Returns:
            - A dictionary representation of the given config.yml file
        """
        if isinstance(config, str):
            with open(config, "r") as ymlfile:
                config = yaml.load(ymlfile, Loader=yaml.Loader)
        return config

    def _open_ds_dict(
        self, ds_dict: Optional[Union[datasets.DatasetDict, str]] = None
    ) -> dict:
        """
        Params:
            - ds_dict: Optionally takes either a datasets.DatasetDict object or a path to a
            - saved datasets.DatasetDict file

        If passed a saved datasets.DatasetDict file path, will read its contents
        If not passed an argument, will load the datasets.DatasetDict from the file path specified
            in the config file
        Cleans (removes stopwords, removes punctuation) the given datasets.DatasetDict object
        based on the config file passed to self

        Returns:
            - A datasets.DatasetDict object with cleaned text
        """

        if isinstance(ds_dict, str):
            ds_dict = datasets.load_from_disk(ds_dict)
        elif ds_dict is None:
            ds_dict = datasets.load_from_disk(self.config["data_path"])

        ds_dict["train"] = ds_dict["train"].map(self._do_preprocessing)
        return ds_dict

    def _clean_up_text(self, line: str) -> str:
        """
        Params:
            - line: a string of text

        Given a string and a set of stop words, removes extraneous punctuation,
        numbers, and the stop words from the string.

        Returns
            - the cleaned up string.
        """
        line = line + " "

        if self.config["remove_punc"]:
            line = re.sub(r",|;|\"|\'|\?|:|!", "", line)  # removes punctuation
            line = re.sub(r"\\", "", line)  # removes punctuation
            line = re.sub(r"\.\s", " ", line)  # removes punctuation

        if self.config["remove_stop_words"]:
            stop_words = (
                self.TAM_STOP_WORDS
                if self.config["lang"] == "tam"
                else self.MAL_STOP_WORDS
            )
            line = " ".join(
                [word for word in line.split() if word not in stop_words]
            )  # removes stop words

        if self.config["remove_num"]:
            line = re.sub(
                r"\s\S*?\d\S*?\s", " ", line
            )  # removes words that contain numbers
            line = re.sub(r"\d", "", line)  # removes numbers

        # line = re.sub(r"[a-zA-Z]", " ", line)             # removes single letters

        return line.strip()

    def _do_preprocessing(self, ds_dict: datasets.DatasetDict) -> datasets.DatasetDict:
        """
        Params
            - ds_dict: Takes in a huggingface object
            - config: Takes in a config dictionary

        Performs preprocessing on all text data. Cleans text data by removing stopwords
        and/or punctuation based on values from the config dictionary.

        Returns
            - a datasetDict with cleaned text
        """
        if (
            self.config["remove_punc"]
            or self.config["remove_stop_words"]
            or self.config["remove_num"]
        ):
            ds_dict["text"] = self._clean_up_text(ds_dict["text"])
        return ds_dict

    def _tfidf_vectors(self) -> spmatrix:
        """
        Return:
            - document-tfidf vectors using sklearns TfidfVectorizer as a spmatrix
        """
        text = np.array(self.ds_dict["train"]["text"])

        tf_idf = TfidfVectorizer(
            input="content",
            encoding="utf-8",
            ngram_range=(1, 3),
            binary=True,
            smooth_idf=False,
        )
        tfidf_vectors = tf_idf.fit_transform(text)

        return tfidf_vectors

    def vec_type(self, vec_type: Optional[str] = None) -> str:
        """
        Params:
            - vec_type: Takes an optional type of vectorization, and makes vectors of the specified type
            - Otherwise, will make vectors of the type specified in the config.yml file

        Creates a document vector representation of a datasets.DatasetDict object based on the specified
            type of vectorization

        Returns:
            - Document vectors as a spamatrix

        """
        if vec_type is None:
            vec_type = self.config["vectors"]

        return vec_type

    def create_vectors(self):
        """
        Creates document feature vectors based on the specified type of vectorization

        Return:
            - document feature vectors as a spmatrix
        """
        if self.vec_type == "tfidf":
            # Calculate TF-IDF Vectors
            vectors = self._tfidf_vectors()

        # TODO: add more vectorization methods here
        # elif self.vec_type == "...":
        else:  # otherwise not a known vectorization method
            raise ValueError(
                f"config argument {self.config['vectors']} not a known vectorization method"
            )

        return vectors

    def get_vectors(self) -> spmatrix:
        """
        Getter method,
        Return:
            - the features vectors as a spmatrix
        """
        return self.vectors


def main():
    # for testing purposes
    config_file = sys.argv[1]
    with open(config_file, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(config["data_path"])

    # ds_dict["train"] = ds_dict["train"].map(do_preprocessing,
    #                                         fn_kwargs={"config": config}
    #                                         )

    # vectors = Vectors(config=config, ds_dict=ds_dict, vec_type="tfidf")
    vectors = Vectors(config_file)
    tfidf_vectors = vectors.get_vectors()

    print(tfidf_vectors)


if __name__ == "__main__":
    main()
