import os
import sys
import yaml

import datasets
from sklearn.model_selection import StratifiedKFold

from classifiers import NaiveBayesClassifier, TransformerLayerVectorClassifier

# from transformer_lm.finetune_transformer import rebalance_ds
from common import GLOBAL_SEED

config_file = sys.argv[1]
with open(config_file, "r") as ymlfile:
    config = yaml.load(ymlfile, Loader=yaml.Loader)

# clf = TransformerLayerVectorClassifier(config=config, lm_name_or_path="xlm-roberta-base", strategy="last4")
clf = TransformerLayerVectorClassifier(config=config, lm_name_or_path="xlm-roberta-large", strategy="last4")
ds_dict: datasets.DatasetDict = datasets.load_from_disk(sys.argv[2])
ds = ds_dict["train"]

skfolds = StratifiedKFold(n_splits=4)

for n, (train_idxs, eval_idxs) in enumerate(
        skfolds.split(range(ds.num_rows), ds["label"])
):
    # if n > 0:
    #     sys.exit(0)
    # train_ds = rebalance_ds(ds.select(train_idxs), seed=GLOBAL_SEED, shuffle=True, shuffle_seed=GLOBAL_SEED)
    train_ds = ds.select(train_idxs)
    eval_ds = ds.select(eval_idxs)
    clf.train(train_ds)
    print(f"train_idxs: {train_idxs}")
    print(f"eval_idxs: {eval_idxs}")
    y_true = eval_ds["label"]
    y_pred = clf.predict(eval_ds)
    print(f"y_true: {y_true}")
    print(f"y_pred: {list(y_pred)}")

# 
# dataset_path = os.path.abspath(config["data_path"])
# 
# ds_dict: datasets.DatasetDict = datasets.load_from_disk(dataset_path)
# 
# classifier = NaiveBayesClassifier(config=config, ds_dict=ds_dict)
# 
# classifier.kfold_validation()
