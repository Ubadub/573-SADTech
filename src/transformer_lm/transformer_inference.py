import os
import sys
from typing import Optional

import datasets

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold

from transformers import (
    AutoTokenizer,
    pipeline,
)

from common import CLASS_LABELS, GLOBAL_SEED, N_FOLDS

# BASE_MODEL_PATH = "outputs/D2/transformer_model/"
TOKENIZER_KWARGS = {"padding": True, "truncation": True, "max_length": 512}


def _infer_map(pipe, **pipe_kwargs):
    def _(x):
        x["inferred_label"] = pipe(x["text"], **pipe_kwargs)
        return x

    return _


def infer(
    lang: str,
    dataset_dict_path: Optional[str] = None,
    model_base_path: Optional[str] = None,
):
    if dataset_dict_path is None:
        dataset_dict_path = os.path.abspath(f"../data/{lang}/train_dataset_dict")
    if model_base_path is None:
        model_base_path = os.path.abspath(f"../outputs/D2/{lang}/transformer_model")

    ds_dict = datasets.load_from_disk(dataset_dict_path)
    ds = ds_dict["train"]

    y_true_pooled = []
    y_pred_pooled = []

    skfolds = StratifiedKFold(4)
    for n, (_, eval_idxs) in enumerate(skfolds.split(range(ds.num_rows), ds["label"])):
        print(f"#### FOLD {n} ####")
        eval_ds = ds.select(eval_idxs)
        fold_model_path = os.path.abspath(os.path.join(model_base_path, f"fold_{n}"))
        tokenizer = AutoTokenizer.from_pretrained(fold_model_path)
        pipe = pipeline(
            task="sentiment-analysis",
            model=fold_model_path,
            tokenizer=tokenizer,
        )
        infer_ds = eval_ds.map(
            _infer_map(pipe, **TOKENIZER_KWARGS)
        )  # .datasets.Value("float64")
        y_true = infer_ds["label"]
        y_pred = [
            CLASS_LABELS.str2int(x[0]["label"]) for x in infer_ds["inferred_label"]
        ]
        y_true_pooled.extend(y_true)
        y_pred_pooled.extend(y_pred)
        print(
            classification_report(
                y_true,
                y_pred,
                labels=range(CLASS_LABELS.num_classes),
                target_names=CLASS_LABELS.names,
                zero_division=0,
            )
        )
        print("\n")

    print(f"#### POOLED SCORES ####")
    print(
        classification_report(
            y_true_pooled,
            y_pred_pooled,
            labels=range(CLASS_LABELS.num_classes),
            target_names=CLASS_LABELS.names,
            zero_division=0,
        )
    )
