"""
    Multinomial Naive Bayes Classifier
"""

import sys
import yaml
import datasets
import numpy as np

from typing import *

from sklearn.naive_bayes import MultinomialNB

from classifier import Classifier


class NaiveBayesClassifier(Classifier):


    def __init__(self, config: dict, ds_train: datasets.DatasetDict):
        super().__init__(
            config=config,
            ds_train=ds_train
        )
        self.model = MultinomialNB(fit_prior=True)


    def train_predict(self, train_indices: np.array = None, dev_indices: np.array = None):
        train_vectors = self.feature_vectors[train_indices]
        dev_vectors = self.feature_vectors[dev_indices]

        train_gold_labels = self.gold_labels[train_indices]
        dev_gold_labels = self.gold_labels[dev_indices]

        self.model.fit(train_vectors, train_gold_labels)
        print("i've fitted")

        predicted = self.model.predict(np.array(dev_vectors))

        return dev_gold_labels, predicted


if __name__ == '__main__':
    # for testing purposes
    config_file = sys.argv[1]
    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    ds_train: datasets.DatasetDict = datasets.load_from_disk(config["data_path"])

    classifier = NaiveBayesClassifier(config=config, ds_train=ds_train)

    classifier.kfold_validation()
