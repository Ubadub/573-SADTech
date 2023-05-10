import os
import sys
import yaml

import datasets

from classifiers import NaiveBayesClassifier, LogisticRegressionClassifier, StochasticGradientDescentClassifier

config_file = sys.argv[1]
with open(config_file, "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

dataset_path = os.path.abspath(config["data_path"])

ds_dict: datasets.DatasetDict = datasets.load_from_disk(dataset_path)

cl_model = config["classifier"]

if cl_model == "nb":
    classifier = NaiveBayesClassifier(config=config, ds_dict=ds_dict)
elif cl_model == "lr":
    classifier = LogisticRegressionClassifier(config=config, ds_dict=ds_dict)
elif cl_model == "sgd":
    classifier = StochasticGradientDescentClassifier(config=config, ds_dict=ds_dict)
else:
    raise ValueError(f"Needs implemented classifier but got {cl_model} instead.")

classifier.kfold_validation()
