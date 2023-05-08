from argparse import ArgumentParser

# import sys
import os

from imblearn.pipeline import Pipeline

import numpy as np

from datasets import DatasetDict, load_from_disk

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import yaml

from common import GLOBAL_SEED

np.random.seed(GLOBAL_SEED)

CFG_CLASSIFIER_CACHE = "ClassifierCachePath"

CFG_TEXT_TRANSFORMERS = "TextTransformers"
CFG_AUDIO_TRANSFORMERS = "AudioTransformers"
CFG_CLASSIFIER = "Classifier"
CFG_RESAMPLERS = "Resamplers"
CFG_NAME = "name"
CFG_CLASS = "class"
CFG_ARGS = "args"
CFG_KWARGS = "kwargs"

TEXT_COL = "text"
AUDIO_COL = "audio"
Y_COL = "label"
# ALL_FEATS = [AUDIO_COL, TEXT_COL]
ALL_FEATS = [TEXT_COL]

PARSER_CONFIG = {
    # "prog": "python -m transformer_lm",
    "description": "For testing",
}

# SUBPARSERS_CONFIG = {
#     "title": "Action",
#     "description": "Action to execute- finetuning or inference.",
#     # "dest": "action",
#     "required": True,
#     "help": "Select finetune to do finetuning or infer to do inference.",
#     # "metavar": f"[{', '.join(filters.CLI_FILTERS.keys())}]",
# }

parser = ArgumentParser(**PARSER_CONFIG)
parser.add_argument(
    "-c",
    "--config",
    required=True,
    # default="../config/config.yml",
    dest="config_path",
    metavar="PATH/TO/CONFIG.YML",
    help="Path to the config YAML file.",
)

parser.add_argument(
    "-d",
    "--dataset",
    required=True,
    # default="../../data/train_dataset_dict",
    dest="dataset_path",
    metavar="PATH/TO/DATASET/",
    help="Path to the dataset.",
)

args = parser.parse_args()

with open(args.config_path, "r") as ymlfile:
    cfg = yaml.unsafe_load(ymlfile)

default_cache_path = f"sklearn_cache/{os.path.basename(args.config_path)}/classifier"
classifier_cache = cfg.get(CFG_CLASSIFIER_CACHE, default_cache_path)

ds_dict: DatasetDict = load_from_disk(args.dataset_path)
ds = ds_dict["train"]

text_transformers_cfg = cfg.get(CFG_TEXT_TRANSFORMERS, [])
text_transformer = Pipeline(
    steps=[
        (
            tr[CFG_NAME],
            tr[CFG_CLASS](*tr.get(CFG_ARGS, []), **tr.get(CFG_KWARGS, {})),
        )
        for tr in text_transformers_cfg
    ],
    # memory="sklearn_cache/text_transformer",
)

print("Text transformer:", text_transformer)

preprocessor = ColumnTransformer(
    transformers=[
        ("text", text_transformer, TEXT_COL),
        # ("audio", audio_transformer, AUDIO_COL),
    ],
    n_jobs=-1,
)

print("Preprocessor:", preprocessor)

resamplers = [
    (
        resampler_cfg[CFG_NAME],
        resampler_cfg[CFG_CLASS](
            *resampler_cfg.get(CFG_ARGS, []),
            **resampler_cfg.get(CFG_KWARGS, {}),
        ),
    )
    for resampler_cfg in cfg.get(CFG_RESAMPLERS, [])
]

print("Resamplers:", resamplers)

classifier_cfg = cfg.get(CFG_CLASSIFIER, {})
classifier = (
    "classifier",
    classifier_cfg[CFG_CLASS](
        *classifier_cfg.get(CFG_ARGS, []),
        **classifier_cfg.get(CFG_KWARGS, {}),
    ),
)

print("Classifier:", classifier)

clf = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        *resamplers,
        classifier,
    ],
    memory=classifier_cache,
    # ("classifier", LogisticRegression())]
)

print("Pipeline:", clf)

skfolds = StratifiedKFold(n_splits=4)

y_true_pooled = []
y_pred_pooled = []

for n, (train_idxs, eval_idxs) in enumerate(
    skfolds.split(range(ds.num_rows), ds[Y_COL])
):
    # if n > 0:
    #     sys.exit(0)
    print(f"train_idxs: {train_idxs}")
    print(f"eval_idxs: {eval_idxs}")
    # train_ds = rebalance_ds(ds.select(train_idxs), seed=GLOBAL_SEED, shuffle=True, shuffle_seed=GLOBAL_SEED)
    train_ds = ds.select(train_idxs)
    eval_ds = ds.select(eval_idxs)
    train_df = train_ds.to_pandas()
    eval_df = eval_ds.to_pandas()
    X_train, y_train = train_df[ALL_FEATS], train_df[Y_COL]
    X_eval, y_true = eval_df[ALL_FEATS], eval_df[Y_COL].to_numpy()

    # clf.fit(train_ds, y_train)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_eval)

    y_true_pooled.extend(y_true)
    y_pred_pooled.extend(y_pred)
    print(f"y_true: {list(y_true)}")
    print(f"y_pred: {list(y_pred)}")

print(classification_report(y_true_pooled, y_pred_pooled))
