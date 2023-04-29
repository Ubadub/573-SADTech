import os
import sys
import yaml

import datasets

from classifiers import NaiveBayesClassifier

# for testing purposes
config_file = sys.argv[1]
with open(config_file, 'r') as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

dataset_path = os.path.abspath(config["data_path"])

ds_train: datasets.DatasetDict = datasets.load_from_disk(dataset_path)

classifier = NaiveBayesClassifier(config=config, ds_train=ds_train)

classifier.kfold_validation()
