import logging
import os
import pickle
import sys
from typing import Optional, Sequence, Union

from datasets import Dataset, DatasetDict, load_from_disk
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import submitit
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted

from pipeline_transformers.utils import (
    assemble_pipeline,
    clean_path,
    exception_handler,
    setup_rng,
)

# CFG_MODEL_SAVE_PATH = "model_save_path"

CFG_MODEL_DIR = "model_dir"
CFG_RESULTS_FILE = "results_file"

Y_COL = "label"

log = logging.getLogger(__name__)


# def crossfold(clf: BaseEstimator, ds: Dataset, do_fit=True, saved_models_root_path=None):
def crossfold(
    clf_or_saved_path: Union[str, BaseEstimator],
    train_ds: Dataset,
    test_ds: Optional[Dataset] = None,
    n_splits: int = 4,
    # feat_cols: Sequence[str],
    y_col: str = Y_COL,
    model_dir: Optional[str] = None,
) -> dict:
    # if not do_fit and saved_models_root_path is None:
    #     sys.exit("Need to refit models or pass path to existing models.")

    if isinstance(clf_or_saved_path, str):
        do_fit = False
    elif isinstance(clf_or_saved_path, BaseEstimator):
        log.info(
            "Pipeline already fitted. Doing crossvalidation inference + test inference."
        )
        do_fit = True
    else:
        log.critical("Need to provide classifier or path to saved classifiers.")
        sys.exit()

    if model_dir is not None:
        if model_dir:  # model_dir could be empty string, i.e. current path
            os.makedirs(model_dir, exist_ok=True)

    skfolds = StratifiedKFold(n_splits=n_splits)

    y_true_pooled = []
    y_pred_pooled = []
    y_scores_pooled = []

    for n, (train_idxs, eval_idxs) in enumerate(
        skfolds.split(range(train_ds.num_rows), train_ds[y_col])
    ):
        # if n > 0:
        #     sys.exit(0)
        log.debug(f"train_idxs: {train_idxs}")
        log.debug(f"eval_idxs: {eval_idxs}")
        # fold_train_ds = rebalance_ds(ds.select(train_idxs), seed=GLOBAL_SEED, shuffle=True, shuffle_seed=GLOBAL_SEED)
        fold_train_ds = train_ds.select(train_idxs)
        eval_ds = train_ds.select(eval_idxs)
        train_df = fold_train_ds.to_pandas()
        eval_df = eval_ds.to_pandas()
        X_train, y_train = train_df, train_df[y_col]
        X_eval, y_true = eval_df, eval_df[y_col].to_numpy()
        # X_train, y_train = train_df[feat_cols], train_df[y_col]
        # X_eval, y_true = eval_df[feat_cols], eval_df[y_col].to_numpy()

        if do_fit:
            # log.debug(f"Fitting with feat_cols: {feat_cols}")
            clf = clf_or_saved_path
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
            clf.fit(X_train, y_train)

        else:
            in_path = os.path.join(clf_or_saved_path, f"fold{n}.model")
            # log.info(f"Loading model from: {in_path}")
            with open(in_path, "rb") as f:
                clf = pickle.load(f)

        if model_dir is not None:
            model_fpath = os.path.join(model_dir, f"fold{n}.model")
            with open(model_fpath, "wb") as f:
                pickle.dump(clf, file=f)
                log.info(f"Pickled pipeline to {model_fpath}")
        # yield clf

        # y_pred = clf.predict(X_eval)

        y_true_pooled.extend(y_true)
        y_pred_pooled.extend(clf.predict(X_eval))
        y_scores_pooled.extend(clf.predict_proba(X_eval))
        log.info(f"Finished fold {n}")

    log.info(
        f"Crossvalidation: \n{classification_report(y_true_pooled, y_pred_pooled)}"
    )

    res_dict = {
        "y_true": y_true_pooled,
        "y_pred": y_pred_pooled,
        "y_scores": y_scores_pooled,
    }
    if test_ds is not None:
        train_df = train_ds.to_pandas()
        test_df = test_ds.to_pandas()
        X_train, y_train = train_df, train_df[y_col]
        X_test, y_true = test_df, test_df[y_col].to_numpy()
        if do_fit:
            clf = clf_or_saved_path
            clf.fit(X_train, y_train)
        else:
            in_path = os.path.join(clf_or_saved_path, f"full.model")
            # log.info(f"Loading model from: {in_path}")
            with open(in_path, "rb") as f:
                clf = pickle.load(f)
        y_pred = clf.predict(X_test)
        y_scores = clf.predict_proba(X_test)

        res_dict["full_y_true"] = y_true
        res_dict["full_y_pred"] = y_pred
        res_dict["full_y_scores"] = y_scores

        log.info(f"Full model: \n{classification_report(y_true, y_pred)}")
        if model_dir is not None:
            model_fpath = os.path.join(model_dir, f"full.model")
            with open(model_fpath, "wb") as f:
                pickle.dump(clf, file=f)
                log.info(f"Pickled full pipeline to {model_fpath}")

    return res_dict


def train(
    pipeline_cfg: DictConfig,
    train_ds: Dataset,
    test_ds: Optional[Dataset] = None,
    n_splits: int = 4,
    model_dir: Optional[str] = None,
    results_file: Optional[str] = None,
    # model_save_path = pipeline_cfg.get(CFG_MODEL_SAVE_PATH)
) -> None:
    clf = assemble_pipeline(pipeline_cfg)
    results = crossfold(
        clf_or_saved_path=clf,
        train_ds=train_ds,
        test_ds=test_ds,
        n_splits=n_splits,
        model_dir=model_dir,
        # feat_cols=[TEXT_COL, pipeline_cfg.audio_col],
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
    log.info(f"override_dirname: {HydraConfig.get().job.override_dirname}")
    log.info(f"cleaned_override_dirname: {cfg.cleaned_override_dirname}")
    log.info(
        f"hydra.sweep.dir/subdir: {HydraConfig.get().sweep.dir}/{HydraConfig.get().sweep.subdir}"
    )
    # if cfg.debug:
    #     log.setLevel(logging.DEBUG)
    # else:
    #     log.setLevel(logging.WARN)
    log.info(OmegaConf.to_yaml(cfg, resolve=False))
    # log.info(
    #     f"STARTING process ID {os.getpid()}."
    #     f"\n\tClassifier: {cfg.pipeline.classifier._target_}."
    #     "\n\tUsing:"
    #     f"\n\t\t{cfg.pipeline.resamplers.keys()}"
    #     f"\n\t\t{cfg.pipeline.postresample_transformers.keys()}."
    #     f"\n\tSeed: {cfg.global_rng.seed}\n\tEnvironment: {submitit.JobEnvironment()}"
    # )
    # log.info(
    #     f"STARTING process ID {os.getpid()}.\n\tClassifier: {cfg.pipeline.classifier._target_}.\n\tUsing:\n\t\t{cfg.pipeline.resamplers.keys()}\n\t\t{cfg.pipeline.postresample_transformers.keys()}.\n\tSeed: {cfg.global_rng.seed}\n\tEnvironment: {submitit.JobEnvironment()}"
    # )
    log.info(
        f"""STARTING process ID {os.getpid()}.
            \n\tClassifier: {cfg.pipeline.classifier._target_}.
            \n\tUsing:
            \n\t\t{cfg.pipeline.resamplers.keys()}.
            \n\t\t{cfg.pipeline.postresample_transformers.keys()}.
            \n\tSeed: {cfg.global_rng.seed}.
            \n\tEnvironment: {submitit.JobEnvironment()}"""
    )
    # log.debug(OmegaConf.to_yaml(cfg, resolve=False))
    # log.debug(OmegaConf.to_yaml(cfg, resolve=True))
    cfg = setup_rng(cfg)
    # log.debug(cfg.global_rng is cfg.pipeline.classifier.random_state)
    # assert cfg.global_rng is cfg.pipeline.classifier.random_state
    # log.debug(OmegaConf.to_yaml(cfg))
    log.debug(OmegaConf.to_yaml(cfg))

    # Path to saved datasets
    ds_path: str = hydra.utils.to_absolute_path(cfg.dataset)

    ds_dict: DatasetDict = load_from_disk(ds_path)
    train_ds: Dataset = ds_dict["train"]
    test_ds: Dataset = ds_dict["test"]

    # train(cfg, ds, model_save_path=cfg.model_save_path)

    pipeline_cfg = hydra.utils.instantiate(cfg.pipeline)  # , _recursive_=False)
    # if "_target_" in pipeline_cfg: # do inference only
    if not isinstance(pipeline_cfg, BaseEstimator):
        # do training
        log.info("Training.")
        train(
            pipeline_cfg=pipeline_cfg,
            train_ds=train_ds,
            test_ds=test_ds,
            n_splits=cfg.n_splits,
            model_dir=cfg.get(CFG_MODEL_DIR),
            results_file=cfg.get(CFG_RESULTS_FILE),
        )
    elif check_is_fitted(pipeline_cfg):  # do inference only
        log.info("Inference.")
        # Path where the saved per-fold classifiers are.
        results = crossfold(
            clf_or_saved_path=pipeline_cfg,
            train_ds=train_ds,
            test_ds=test_ds,
            n_splits=cfg.n_splits,
            # model_dir=model_dir,
            model_dir=cfg.get(CFG_MODEL_DIR),
        )
        # model_save_path: str = hydra.utils.to_absolute_path(
        #     pipeline_cfg.model_save_path
        # )
    else:  # TODO?
        log.critical(
            "Presently, directly instantiated pipelines must already be fitted"
        )
        sys.exit()


if __name__ == "__main__":
    main()
