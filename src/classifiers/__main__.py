import os
import sys
import yaml

import datasets

from classifiers import NaiveBayesClassifier

config_file = sys.argv[1]
with open(config_file, "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

dataset_path = os.path.abspath(config["data_path"])

ds_dict: datasets.DatasetDict = datasets.load_from_disk(dataset_path)

classifier = NaiveBayesClassifier(config=config, ds_dict=ds_dict)

classifier.kfold_validation()
