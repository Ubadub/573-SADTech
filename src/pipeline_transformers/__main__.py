from argparse import ArgumentParser

import os
import pickle
import sys
from typing import Optional, Union

from imblearn.pipeline import Pipeline

from datasets import Dataset, DatasetDict, load_from_disk

import hydra

import numpy as np

from omegaconf import DictConfig, OmegaConf

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

# from common import GLOBAL_SEED

from pipeline_transformers.utils import setup_rng

# np.random.seed(GLOBAL_SEED)

# from . import get_or_create_rng


CFG_MODEL_SAVE_PATH = "model_save_path"

CFG_CLASSIFIER_CACHE = "ClassifierCachePath"

CFG_TEXT_TRANSFORMERS = "TextTransformers"
CFG_AUDIO_TRANSFORMERS = "AudioTransformers"
CFG_TRANSFORMERS = "transformers"
CFG_CLASSIFIER = "Classifier"
CFG_PRERESAMPLE_TRANSFORMERS = "preresample_transformers"
CFG_RESAMPLERS = "resamplers"
CFG_POSTRESAMPLE_TRANSFORMERS = "postresample_transformers"
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
def crossfold(clf_or_saved_path: Union[str, BaseEstimator], ds: Dataset, n_splits: int):
    # if not do_fit and saved_models_root_path is None:
    #     sys.exit("Need to refit models or pass path to existing models.")

    if isinstance(clf_or_saved_path, str):
        do_fit = False
    elif isinstance(clf_or_saved_path, BaseEstimator):
        do_fit = True
    else:
        sys.exit("Need to provide classifier or path to saved classifiers.")

    skfolds = StratifiedKFold(n_splits=n_splits)

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


def train(
    pipeline_cfg: DictConfig,
    ds: Dataset,
    n_splits: int,
    # default_cache_path = f"sklearn_cache/{os.path.basename(cfg_path)}/classifier"
    # classifier_cache = cfg.get(CFG_CLASSIFIER_CACHE, default_cache_path)
    # model_save_path = pipeline_cfg.get(CFG_MODEL_SAVE_PATH)
):
    text_preprocessors = pipeline_cfg.text
    audio_preprocessors = pipeline_cfg.audio

    # text_transformers_cfg = pipeline_cfg.get(CFG_TEXT_TRANSFORMERS, [])
    text_transformer = Pipeline(
        steps=[
            ("vectorizer", text_preprocessors.vectorizer),  # mandatory
            *(
                (tr_name, tr)
                for tr_name, tr in text_preprocessors.get(CFG_TRANSFORMERS, {}).items()
            ),
        ],
        # memory="sklearn_cache/text_transformer",
    )

    # audio_transformer = Pipeline(
    #     steps=[
    #         audio_preprocessors.vectorizer,  # mandatory
    #         *(
    #             (tr_name, tr)
    #             for tr_name, tr in audio_preprocessors.get(CFG_TRANSFORMERS, {}).items()
    #         ),
    #     ],
    # )

    print("Text transformer:", text_transformer)
    # print("Audio transformer:", audio_transformer)

    preprocessor = ColumnTransformer(
        transformers=[
            (TEXT_COL, text_transformer, TEXT_COL),
            # (AUDIO_COL, audio_transformer, AUDIO_COL),
        ],
        n_jobs=-1,
    )

    print("Preprocessor:", preprocessor)

    preresample_transformers = [
        (tr_name, tr)
        for tr_name, tr in pipeline_cfg.get(CFG_PRERESAMPLE_TRANSFORMERS, {}).items()
    ]

    print("Preresample transformers:", preresample_transformers)

    resamplers = [
        (resampler_name, resampler)
        for resampler_name, resampler in pipeline_cfg.get(CFG_RESAMPLERS, {}).items()
    ]

    print("Resamplers:", resamplers)

    postresample_transformers = [
        (tr_name, tr)
        for tr_name, tr in pipeline_cfg.get(CFG_POSTRESAMPLE_TRANSFORMERS, {}).items()
    ]

    print("Postresample transformers:", postresample_transformers)

    classifier = pipeline_cfg.classifier  # mandatory

    print("Classifier:", classifier)

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            *preresample_transformers,
            *resamplers,
            *postresample_transformers,
            ("classifier", classifier),
        ],
        # memory=classifier_cache,
    )

    print("Pipeline:", clf)
    for n, fitted_clf in enumerate(
        crossfold(clf_or_saved_path=clf, ds=ds, n_splits=n_splits)
    ):
        print(f"Finished fold {n}")
    #     if model_save_path is not None:
    #         out_path = os.path.join(model_save_path, f"{n}.pkl")
    #         with open(out_path, "wb") as f:
    #             pickle.dump(fitted_clf, f)
    #             print("Saved model to:", out_path)


def infer(ds: Dataset, saved_models_dir: str):
    for _ in crossfold(clf_or_saved_path=saved_models_dir, ds=ds):
        pass
    # crossfold(clf_or_path=saved_models_dir, ds=ds, do_fit=True)


# @hydra.main( version_base=None, config_path="../config/hydra_root", config_name="config")
@hydra.main(version_base=None, config_path="../config/hydra_root")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=False))
    # print(OmegaConf.to_yaml(cfg, resolve=True))
    cfg = setup_rng(cfg)
    # print(cfg.global_rng is cfg.pipeline.classifier.random_state)
    # assert cfg.global_rng is cfg.pipeline.classifier.random_state
    # print(OmegaConf.to_yaml(cfg))

    is_debug = cfg.debug

    # Path to saved datasets
    ds_path: str = hydra.utils.to_absolute_path(cfg.dataset)

    ds_dict: DatasetDict = load_from_disk(ds_path)
    ds: Dataset = ds_dict["train"]

    # train(cfg, ds, model_save_path=cfg.model_save_path)

    pipeline_cfg = hydra.utils.instantiate(cfg.pipeline)  # , _recursive_=False)
    # if "_target_" in pipeline_cfg: # do inference only
    if isinstance(pipeline_cfg, BaseEstimator) and check_is_fitted(
        pipeline_cfg
    ):  # do inference only
        print("Inference.")
        # Path where the saved per-fold classifiers are.
        model_save_path: str = hydra.utils.to_absolute_path(
            pipeline_cfg.model_save_path
        )
    elif isinstance(pipeline_cfg, BaseEstimator):  # TODO
        sys.exit("Presently, directly instantiated pipelines must already be fitted")
    else:
        # do training
        print("Training.")
        train(pipeline_cfg, ds, cfg.n_splits)

    # args.func(**func_kwargs)


if __name__ == "__main__":
    main()
