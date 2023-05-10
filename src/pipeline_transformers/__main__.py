from argparse import ArgumentParser

import os
import pickle
import sys
from typing import Optional, Union

from imblearn.pipeline import Pipeline

import numpy as np

from datasets import Dataset, DatasetDict, load_from_disk

from sklearn.base import BaseEstimator
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
    "description": "For testing/deliverables",
}

SUBPARSERS_CONFIG = {
    "title": "Action",
    "description": "Action to execute- training or inference.",
    # "dest": "action",
    "required": True,
    "help": "Select finetune to do finetuning or infer to do inference.",
    # "metavar": f"[{', '.join(filters.CLI_FILTERS.keys())}]",
}


# def crossfold(clf: BaseEstimator, ds: Dataset, do_fit=True, saved_models_root_path=None):
def crossfold(clf_or_saved_path: Union[str, BaseEstimator], ds: Dataset):
    # if not do_fit and saved_models_root_path is None:
    #     sys.exit("Need to refit models or pass path to existing models.")

    if isinstance(clf_or_saved_path, str):
        do_fit = False
    elif isinstance(clf_or_saved_path, BaseEstimator):
        do_fit = True
    else:
        sys.exit("Need to provide classifier or path to saved classifiers.")

    skfolds = StratifiedKFold(n_splits=4)

    y_true_pooled = []
    y_pred_pooled = []

    for n, (train_idxs, eval_idxs) in enumerate(
        skfolds.split(range(ds.num_rows), ds[Y_COL])
    ):
        # if n > 0:
        #     sys.exit(0)
        # print(f"train_idxs: {train_idxs}")
        # print(f"eval_idxs: {eval_idxs}")
        # train_ds = rebalance_ds(ds.select(train_idxs), seed=GLOBAL_SEED, shuffle=True, shuffle_seed=GLOBAL_SEED)
        train_ds = ds.select(train_idxs)
        eval_ds = ds.select(eval_idxs)
        train_df = train_ds.to_pandas()
        eval_df = eval_ds.to_pandas()
        X_train, y_train = train_df[ALL_FEATS], train_df[Y_COL]
        X_eval, y_true = eval_df[ALL_FEATS], eval_df[Y_COL].to_numpy()

        if do_fit:
            clf = clf_or_saved_path
            # clf.fit(train_ds, y_train)
            clf.fit(X_train, y_train)
        else:
            in_path = os.path.join(clf_or_saved_path, f"{n}.pkl")
            # print("Loading model from:", in_path)
            with open(in_path, "rb") as f:
                clf = pickle.load(f)

        yield clf

        y_pred = clf.predict(X_eval)

        y_true_pooled.extend(y_true)
        y_pred_pooled.extend(y_pred)
        # print(f"y_true: {list(y_true)}")
        # print(f"y_pred: {list(y_pred)}")

    print(classification_report(y_true_pooled, y_pred_pooled))


def train(cfg_path: str, dataset_path: str, save_model_to: Optional[str] = None):
    with open(cfg_path, "r") as ymlfile:
        cfg = yaml.unsafe_load(ymlfile)

    default_cache_path = f"sklearn_cache/{os.path.basename(cfg_path)}/classifier"
    classifier_cache = cfg.get(CFG_CLASSIFIER_CACHE, default_cache_path)

    ds_dict: DatasetDict = load_from_disk(dataset_path)
    ds: Dataset = ds_dict["train"]

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

    audio_transformers_cfg = cfg.get(CFG_AUDIO_TRANSFORMERS, [])
    audio_transformer = Pipeline(
        steps=[
            (
                tr[CFG_NAME],
                tr[CFG_CLASS](*tr.get(CFG_ARGS, []), **tr.get(CFG_KWARGS, {})),
            )
            for tr in audio_transformers_cfg
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, TEXT_COL),
            ("audio", audio_transformer, AUDIO_COL),
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
        # memory=classifier_cache,
        # ("classifier", LogisticRegression())]
    )

    print("Pipeline:", clf)

    for n, fitted_clf in enumerate(crossfold(clf_or_saved_path=clf, ds=ds)):
        if save_model_to is not None:
            out_path = os.path.join(save_model_to, f"{n}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(fitted_clf, f)
                print("Saved model to:", out_path)


def infer(dataset_path: str, saved_models_dir: str):
    ds_dict: DatasetDict = load_from_disk(dataset_path)
    ds: Dataset = ds_dict["train"]
    for _ in crossfold(clf_or_saved_path=saved_models_dir, ds=ds):
        pass
    # crossfold(clf_or_path=saved_models_dir, ds=ds, do_fit=True)


def main():
    parser = ArgumentParser(**PARSER_CONFIG)
    subparsers = parser.add_subparsers(**SUBPARSERS_CONFIG)

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        # default="../../data/train_dataset_dict",
        dest="dataset_path",
        metavar="PATH/TO/DATASET/",
        help="Path to the dataset.",
    )

    inf_subparser = subparsers.add_parser(
        "infer", aliases=["i", "inf"], description="Do inference"
    )
    inf_subparser.add_argument(
        "-m",
        "--saved_model_path",
        required=True,
        dest="saved_models_dir",
        metavar="PATH/TO/SAVED/CLASSIFIERS",
        help="Path where the saved per-fold classifiers are.",
    )
    # inf_subparser.add_argument(
    #     "-l",
    #     "--lang",
    #     required=True,
    #     choices=["tam", "mal"],
    #     help="Language (tam for Tamil, mal for Malayalam)",
    # )
    inf_subparser.set_defaults(func=infer)

    train_subparser = subparsers.add_parser(
        "train",
        aliases=["t", "tr", "train", "f", "finetune"],
        description="Finetune a pretrained model from HuggingFace",
    )
    train_subparser.add_argument(
        "-c",
        "--config",
        required=True,
        dest="cfg_path",
        metavar="PATH/TO/CONFIG.YML",
        help="Path to the config YAML file for this train run.",
    )
    train_subparser.add_argument(
        "-m",
        "--save_model_to",
        default=None,
        dest="save_model_to",
        metavar="PATH/TO/SAVE/CLASSIFIERS",
        help="Path where to save the per-fold classifiers.",
    )
    train_subparser.set_defaults(func=train)

    args = parser.parse_args()
    func_kwargs = dict(vars(args))
    del func_kwargs["func"]

    args.func(**func_kwargs)


if __name__ == "__main__":
    main()
