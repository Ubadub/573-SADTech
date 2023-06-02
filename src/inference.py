import logging
import os
import pickle

from datasets import Dataset, DatasetDict, load_from_disk
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from common import CLASS_LABELS
from pipeline_transformers.__main__ import crossfold


log = logging.getLogger(__name__)

# def crossfold(n_splits: int, model_dir: str):
#     skfolds = StratifiedKFold(n_splits=n_splits)
#
#     for n, (train_idxs, eval_idxs) in enumerate(
#         skfolds.split(range(len(train_df.index)), train_df[y_col])
#     ):
#         in_path = os.path.join(model_dir, f"fold{n}.model"))
#         clf = load_pipeline(in_path)


def _make_path(cfg: DictConfig, section: str):
    # cfg_section = cfg.get(section)
    root_dir = hydra.utils.to_absolute_path(cfg.root_dir)

    return os.path.join(root_dir, cfg.sub_dir, cfg.get(section), cfg.file_name)


@hydra.main(version_base=None, config_path="config/hydra_root", config_name="inference")
def main(cfg: DictConfig) -> None:
    logging.debug(OmegaConf.to_yaml(cfg, resolve=True))
    ds_path: str = hydra.utils.to_absolute_path(cfg.dataset)

    ds_dict: DatasetDict = load_from_disk(ds_path)
    train_ds: Dataset = ds_dict["train"]
    test_ds: Dataset = ds_dict["test"]
    train_df: pd.DataFrame = train_ds.to_pandas()
    test_df: pd.DataFrame = test_ds.to_pandas()

    n_splits = cfg.n_splits
    results = crossfold(
        do_fit=False,
        clf_or_saved_path=hydra.utils.to_absolute_path(cfg.pipeline_path),
        train_df=train_df,
        test_df=test_df,
        n_splits=cfg.n_splits,
        # feat_cols=[pipeline_cfg.text.column_name, pipeline_cfg.audio_col.column_name],
        save_dir=None,
    )

    dev_results = zip(
        results["dev_files"], results["dev_y_true"], results["dev_y_pred"]
    )
    test_results = zip(
        results["test_files"], results["test_y_true"], results["test_y_pred"]
    )

    for results_grp, subdir in zip(
        (dev_results, test_results), ("dev_subdir", "test_subdir")
    ):
        outputs_file = _make_path(cfg.outputs, subdir)
        results_file = _make_path(cfg.results, subdir)
        # outputs_file = os.path.join(outputs_dir, f"{cfg.lang}.tsv")
        # results_file = os.path.join(results_dir, f"{cfg.lang}.out")

        logging.debug(f"outputs_file: {outputs_file}")
        logging.debug(f"results_file: {results_file}")

        os.makedirs(os.path.dirname(outputs_file), exist_ok=True)
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        y_true = []  # ugly hack
        y_pred = []  # ugly hack
        with open(outputs_file, "w") as outputs_file:
            print("file", "true_label", "predicted_label", sep="\t", file=outputs_file)
            for file_name, y_true_val, y_pred_val in results_grp:
                y_true.append(y_true_val)
                y_pred.append(y_pred_val)
                print(
                    file_name,
                    CLASS_LABELS.int2str(int(y_true_val)),
                    CLASS_LABELS.int2str(int(y_pred_val)),
                    file=outputs_file,
                    sep="\t",
                )
        with open(results_file, "a") as results_file:
            print(
                classification_report(
                    y_true,
                    y_pred,
                    labels=range(CLASS_LABELS.num_classes),
                    target_names=CLASS_LABELS.names,
                    zero_division=0,
                ),
                file=results_file,
                # sep="\n",
            )


if __name__ == "__main__":
    main()
