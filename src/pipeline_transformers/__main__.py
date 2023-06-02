import logging
import os
import pickle
import sys
from typing import Optional, Sequence, Union

from datasets import Dataset, DatasetDict, load_from_disk
import hydra
from hydra.core.hydra_config import HydraConfig
from imblearn.pipeline import Pipeline
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import submitit
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from pipeline_transformers.utils import (
    assemble_pipeline,
    clean_path,
    exception_handler,
    load_pipeline,
    setup_rng,
)

# CFG_MODEL_SAVE_PATH = "model_save_path"

CFG_MODEL_DIR = "model_dir"
CFG_RESULTS_FILE = "results_file"

Y_COL = "label"

log = logging.getLogger(__name__)


def fit_save(
    clf: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_dir: Optional[str] = None,
    save_file_name: str = "model",
) -> Pipeline:
    clf.fit(X_train, y_train)

    if save_dir is not None:
        model_fpath = os.path.join(save_dir, save_file_name)
        with open(model_fpath, "wb") as f:
            pickle.dump(clf, file=f)
            log.info(f"Pickled {save_file_name} pipeline to {model_fpath}.")
    ## X_new = clf.fit_transform(X_train, y_train)
    ## log.debug(f"X_new shape: X_new.shape")
    ## log.debug(f"X_new: X_new")
    ## clf_scaler = clf.named_steps["scaler"]
    ## # clf_preproc = clf.named_steps["preprocessor"]
    ## clf_selector = clf.named_steps["select_from_model"]
    ## support_bool = clf_selector.get_support()
    ## support_idxs = clf_selector.get_support(indices=True)
    ## log.debug(f"clf_selector bool support:\n{support_bool}")
    ## log.debug(f"clf_selector bool support shape:\n{support_bool.shape}")
    ## log.debug(f"clf_selector idxs support:\n{support_idxs}")
    ## log.debug(f"clf_selector idxs support shape:\n{support_idxs.shape}")

    return clf


def fit_save_or_load(
    do_fit: bool,
    clf_or_saved_path: Union[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    save_dir: Optional[str] = None,
    save_file_name: str = "model",
):
    if do_fit:
        clf = fit_save(
            clf_or_saved_path,
            X_train,
            y_train,
            save_dir=save_dir,
            save_file_name=save_file_name,
            # save_file_name=f"fold{n}.model",
        )
    else:
        # in_path = os.path.join(clf_or_saved_path, f"fold{n}.model")
        in_path = os.path.join(clf_or_saved_path, save_file_name)
        clf = load_pipeline(in_path)

    return clf


# def crossfold(clf: BaseEstimator, ds: Dataset, do_fit=True, saved_models_root_path=None):
def crossfold(
    do_fit: bool,
    clf_or_saved_path: Union[str, BaseEstimator],
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    n_splits: int = 4,
    # feat_cols: Sequence[str],
    y_col: str = Y_COL,
    save_dir: Optional[str] = None,
) -> dict:
    # if not do_fit and saved_models_root_path is None:
    #     sys.exit("Need to refit models or pass path to existing models.")
    # if isinstance(clf_or_saved_path, str):
    #     do_fit = False
    # elif isinstance(clf_or_saved_path, BaseEstimator):
    #     log.info(
    #         "Pipeline already fitted. Doing crossvalidation inference + test inference."
    #     )
    #     do_fit = True
    # else:
    #     log.critical("Need to provide classifier or path to saved classifiers.")
    #     sys.exit()

    if save_dir is not None:
        if save_dir:  # save_dir could be empty string, i.e. current path
            os.makedirs(save_dir, exist_ok=True)

    skfolds = StratifiedKFold(n_splits=n_splits)

    y_true_pooled = []
    y_pred_pooled = []
    y_scores_pooled = []
    dev_files_pooled = []

    for n, (train_idxs, eval_idxs) in enumerate(
        skfolds.split(range(len(train_df.index)), train_df[y_col])
    ):
        log.debug(f"train_idxs: {train_idxs}")
        log.debug(f"eval_idxs: {eval_idxs}")
        # fold_train_ds = rebalance_ds(ds.select(train_idxs), seed=GLOBAL_SEED, shuffle=True, shuffle_seed=GLOBAL_SEED)
        # fold_train_ds = train_ds.select(train_idxs)
        # eval_ds = train_ds.select(eval_idxs)
        # train_df = fold_train_ds.to_pandas()
        # eval_df = eval_ds.to_pandas()
        fold_train_df = train_df.iloc[train_idxs].reset_index(drop=True)
        fold_eval_df = train_df.iloc[eval_idxs].reset_index(drop=True)
        X_train, y_train = fold_train_df, fold_train_df[y_col]
        X_eval, y_true = fold_eval_df, fold_eval_df[y_col].to_numpy()

        clf = fit_save_or_load(
            do_fit,
            clf_or_saved_path,
            X_train,
            y_train,
            save_dir=save_dir,
            save_file_name=f"fold{n}.model",
        )

        # if do_fit:
        #     clf = fit_save(
        #         clf_or_saved_path,
        #         X_train,
        #         y_train,
        #         save_dir=save_dir,
        #         save_file_name=f"fold{n}.model",
        #     )
        # else:
        #     in_path = os.path.join(clf_or_saved_path, f"fold{n}.model")
        #     clf = load_pipeline(in_path)

        # yield clf

        dev_files_pooled.extend(fold_eval_df["file"])
        y_true_pooled.extend(y_true)
        y_pred_pooled.extend(clf.predict(X_eval))
        y_scores_pooled.extend(clf.predict_proba(X_eval))
        log.info(f"Finished fold {n}")

    log.info(
        f"Crossvalidation: \n{classification_report(y_true_pooled, y_pred_pooled)}"
    )

    res_dict = {
        "dev_files": dev_files_pooled,
        "dev_y_true": y_true_pooled,
        "dev_y_pred": y_pred_pooled,
        "dev_y_scores": y_scores_pooled,
    }
    if test_df is not None:
        X_train, y_train = train_df, train_df[y_col]
        X_test, y_true = test_df, test_df[y_col].to_numpy()
        clf = fit_save_or_load(
            do_fit,
            clf_or_saved_path,
            X_train,
            y_train,
            save_dir=save_dir,
            save_file_name="full.model",
        )

        y_pred = clf.predict(X_test)
        y_scores = clf.predict_proba(X_test)

        res_dict["test_files"] = X_test["file"].to_list()
        res_dict["test_y_true"] = y_true
        res_dict["test_y_pred"] = y_pred
        res_dict["test_y_scores"] = y_scores

        log.info(f"Full model: \n{classification_report(y_true, y_pred)}")
        # if do_fit:
        #     clf = fit_save(
        #         clf_or_saved_path,
        #         X_train,
        #         y_train,
        #         save_dir=save_dir,
        #         save_file_name=,
        #     )
        #     # clf = fit(clf_or_saved_path, X_train, y_train)
        #     # if save_dir is not None:
        #     #     model_fpath = os.path.join(save_dir, f"full.model")
        #     #     with open(model_fpath, "wb") as f:
        #     #         pickle.dump(clf, file=f)
        #     #         log.info(f"Pickled full pipeline to {model_fpath}")
        # else:
        #     in_path = os.path.join(clf_or_saved_path, "full.model")
        #     clf = load_pipeline(in_path)
        #     # in_path = os.path.join(clf_or_saved_path, f"full.model")
        #     # # log.info(f"Loading model from: {in_path}")
        #     # with open(in_path, "rb") as f:
        #     #     clf = pickle.load(f)

    return res_dict


def train(
    pipeline_cfg: Union[DictConfig, str],
    train_df: pd.DataFrame,
    test_df: Optional[pd.DataFrame] = None,
    # train_ds: Dataset,
    # test_ds: Optional[Dataset] = None,
    n_splits: int = 4,
    save_dir: Optional[str] = None,
    results_file: Optional[str] = None,
    # model_save_path = pipeline_cfg.get(CFG_MODEL_SAVE_PATH)
) -> None:
    # if isinstance(pipeline_cfg is not None:
    if isinstance(pipeline_cfg, DictConfig):
        clf = assemble_pipeline(pipeline_cfg)
        feat_cols = []
        if "text" in pipeline_cfg:
            feat_cols.append(pipeline_cfg.text.column_name)
        if "audio" in pipeline_cfg:
            feat_cols.append(pipeline_cfg.audio.column_name)
        try:
            check_is_fitted(clf)
            do_fit = False
        except NotFittedError:
            do_fit = True

    elif isinstance(pipeline_cfg, str):
        clf = pipeline_cfg
        do_fit = False

    log.debug(
        f"Doing crossfold with do_fit: {do_fit},"
        f"predictor columns: {feat_cols},"
        f"and label column: {Y_COL}."
    )

    results = crossfold(
        do_fit=do_fit,
        clf_or_saved_path=clf,
        train_df=train_df,
        test_df=test_df,
        n_splits=n_splits,
        # feat_cols=[pipeline_cfg.text.column_name, pipeline_cfg.audio_col.column_name],
        save_dir=save_dir,
    )
    if results_file is not None:
        results_dir = os.path.dirname(results_file)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        # out_path = os.path.join(results_file, "results.pkl")
        with open(results_file, "wb") as f:
            pickle.dump(results, file=f)
            log.info(f"Pickled results to {results_file}")

    # for n, fitted_clf in enumerate(
    #     crossfold(clf_or_saved_path=clf, ds=ds, n_splits=n_splits)
    # ):
    #     log.info(f"Finished fold {n}")
    #     if model_save_path is not None:
    #         out_path = os.path.join(model_save_path, f"{n}.pkl")
    #         with open(out_path, "wb") as f:
    #             pickle.dump(fitted_clf, f)
    #             log.info(f"Saved model to: {out_path}")


# def infer(ds: Dataset, saved_models_dir: str):
#     for _ in crossfold(clf_or_saved_path=saved_models_dir, ds=ds):
#         pass
#     # crossfold(clf_or_path=saved_models_dir, ds=ds, do_fit=True)


# @hydra.main(version_base=None, config_path="../config/hydra_root", config_name="config")
@hydra.main(version_base=None, config_path="../config/hydra_root")
@exception_handler(lambda e: log.critical(e, exc_info=True))
def main(cfg: DictConfig) -> None:
    log.debug(f"override_dirname: {HydraConfig.get().job.override_dirname}")
    log.debug(f"cleaned_override_dirname: {cfg.cleaned_override_dirname}")
    log.debug(
        "hydra.sweep.dir/subdir: "
        f"{HydraConfig.get().sweep.dir}/{HydraConfig.get().sweep.subdir}"
    )
    # if cfg.debug:
    #     log.setLevel(logging.DEBUG)
    # else:
    #     log.setLevel(logging.WARN)
    log.debug(OmegaConf.to_yaml(cfg, resolve=False))
    # log.debug(OmegaConf.to_yaml(cfg, resolve=True))
    log.debug(
        f"STARTING process ID {os.getpid()}."
        # "\n\tClassifier: {cfg.pipeline.classifier._target_}."
        # "\n\tUsing:"
        # "\n\t\t{cfg.pipeline.resamplers.keys()}."
        # "\n\t\t{cfg.pipeline.postresample_transformers.keys()}."
        "\n\tSeed: {cfg.global_rng.seed}."
        "\n\tEnvironment: {submitit.JobEnvironment()}"
    )
    cfg = setup_rng(cfg)
    # log.debug(cfg.global_rng is cfg.pipeline.classifier.random_state)
    # assert cfg.global_rng is cfg.pipeline.classifier.random_state

    # Path to saved datasets
    ds_path: str = hydra.utils.to_absolute_path(cfg.dataset)

    ds_dict: DatasetDict = load_from_disk(ds_path)
    train_ds: Dataset = ds_dict["train"]
    test_ds: Dataset = ds_dict["test"]

    train_df: pd.DataFrame = train_ds.to_pandas()
    test_df: pd.DataFrame = test_ds.to_pandas()

    # train(cfg, ds, model_save_path=cfg.model_save_path)

    pipeline_cfg = hydra.utils.instantiate(cfg.pipeline)  # , _recursive_=False)
    log.debug(f"type(pipeline_cfg): {type(pipeline_cfg)}")

    # if "_target_" in pipeline_cfg: # do inference only
    # if not isinstance(pipeline_cfg, BaseEstimator):
    if isinstance(pipeline_cfg, DictConfig):
        # do training
        log.info("Training.")
        train(
            pipeline_cfg=pipeline_cfg,
            train_df=train_df,
            test_df=test_df,
            n_splits=cfg.n_splits,
            save_dir=cfg.get(CFG_MODEL_DIR),
            results_file=cfg.get(CFG_RESULTS_FILE),
        )
    # elif check_is_fitted(pipeline_cfg):  # do inference only
    # Path where the saved per-fold classifiers are.
    elif isinstance(pipeline_cfg, str):  # do inference only
        log.info("Inference.")
        train(
            # do_fit=do_fit,
            pipeline_cfg=pipeline_cfg,
            train_df=train_df,
            test_df=test_df,
            n_splits=cfg.n_splits,
            save_dir=None,
            results_file=hydra.utils.to_absolute_path(cfg.get(CFG_RESULTS_FILE)),
            # save_dir=save_dir,
            # save_dir=cfg.get(CFG_MODEL_DIR),
        )
        # results = crossfold(
        #     do_fit=do_fit,
        #     clf_or_saved_path=cfg.get(CFG_MODEL_DIR),
        #     train_df=train_df,
        #     test_df=test_df,
        #     n_splits=cfg.n_splits,
        #     # save_dir=save_dir,
        #     # save_dir=cfg.get(CFG_MODEL_DIR),
        # )
        # model_save_path: str = hydra.utils.to_absolute_path(
        #     pipeline_cfg.model_save_path
        # )
    else:  # TODO?
        log.critical(
            f"Pipeline must be DictConfig or str, but instead found: {type(pipeline_cfg)}"
            # "Presently, directly instantiated pipelines must already be fitted"
        )
        sys.exit()


if __name__ == "__main__":
    main()
