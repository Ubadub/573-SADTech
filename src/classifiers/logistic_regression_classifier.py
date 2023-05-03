"""
    Logistic Regression Classifier
"""

# from typing import *

import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression

from classifiers import Classifier


class LogisticRegressionClassifier(Classifier):
    def __init__(self, config: dict, ds_dict: datasets.DatasetDict) -> None:
        """
        Params:
            - config: a configuration .yml file
            - ds_dict: a huggingface datasets object containing the train data

        Initializes a Logistic Regression Classifier Model
        """
        super().__init__(config=config, ds_dict=ds_dict)
        self.model = LogisticRegression(multi_class="multinomial")

    def train_predict(
        self, train_indices: np.ndarray, dev_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Params:
            - train_indices: A list of indices corresponding to the files used for training
            - dev_indices: A list of indices corresponding to the files used for evaluating

        This method trains the Logistic Regression Classifier using the given train data, and also predicts the labels
            of the given Development data

        Returns:
            - dev_gold_labels
            - predicted
        """
        train_vectors = self.feature_vectors[train_indices]
        dev_vectors = self.feature_vectors[dev_indices]

        train_gold_labels = self.gold_labels[train_indices]
        dev_gold_labels = self.gold_labels[dev_indices]

        self.model.fit(train_vectors, train_gold_labels)
        predicted = self.model.predict(np.array(dev_vectors))

        return dev_gold_labels, predicted
