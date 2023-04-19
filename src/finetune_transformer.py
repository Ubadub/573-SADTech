import os
import sys
from typing import Any, Callable, Optional, Sequence, Union

import datasets

# from datasets import Features, ClassLabel, Dataset, Value, load_dataset
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import StratifiedKFold
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

# logger = logging.getLogger(__name__)
#
# # Setup logging
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     handlers=[logging.StreamHandler(sys.stdout)],
# )
#
# # set the main code and the modules it uses to the same log-level according to the node
# log_level = training_args.get_process_log_level()
# logger.setLevel(log_level)
# datasets.utils.logging.set_verbosity(log_level)
# transformers.utils.logging.set_verbosity(log_level)

GLOBAL_SEED = 573

N_FOLDS = 2 #4

PRETRAINED_MODEL = "ai4bharat/indic-bert"

CLASS_NAMES = (
    "HIGHLY NEGATIVE",
    "NEGATIVE",
    "NEUTRAL",
    "POSITIVE",
    "HIGHLY POSITIVE",
)

CLASS_LABELS = datasets.ClassLabel(names=CLASS_NAMES)

# FEATS = datasets.Features(
#     {"text": datasets.Value(dtype="string"), "label": CLASS_LABELS}
# )


def process_raw_dataset(
    entry: dict[str, Any],
    class_labels: datasets.ClassLabel,
    path: str = ".",
    ext: str = "txt",
) -> dict[str, Any]:
    fname = f'{entry["file"]}.{ext}'
    fpath = os.path.join(path, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        lines = f.readlines()
        text = "".join(l.strip() for l in lines)
        entry["text"] = text
    entry["label"] = class_labels.str2int(entry["label"].strip().upper())
    del entry["file"]
    return entry


def assemble_dataset(
    lang: str,
    class_labels: datasets.ClassLabel,
    root_data_dir: str = "../data/",
    subdir: str = "text/",
    labels_file_name: str = "all.csv",
    delimiter: str = ",",
) -> datasets.DatasetDict:
    ext = os.path.splitext(labels_file_name)[-1][1:]
    lang_dir = os.path.join(root_data_dir, lang)
    text_file_dir = os.path.join(lang_dir, subdir)
    labels_file = os.path.join(lang_dir, labels_file_name)
    raw_ds = datasets.load_dataset(ext, data_files=labels_file, delimiter=delimiter)
    #    raw_ds = datasets.load_dataset("csv", data_files="../data/tam/all.csv", delimiter=",")
    #    raw_ds = datasets.load_dataset(labels_file.split(".")[-1], )

    ds = raw_ds.map(
        process_raw_dataset,
        fn_kwargs={"path": text_file_dir, "class_labels": class_labels},
    ).cast_column("label", class_labels)
    return ds


def compute_metrics(class_names: Sequence[str]) -> Callable[[EvalPrediction], dict]:
    print("IN HERE")
    def _(eval_pred: EvalPrediction) -> dict:
        print("WAYYYYY IN HERE")
        y_pred, y_true = eval_pred
        y_pred = np.argmax(y_pred, axis=1)
        metrics = classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            output_dict=True,
        )
        return metrics

    return _


#    return accuracy.compute(predictions=predictions, references=labels)


def finetune_for_sequence_classification(
    lang: str,
    pretrained_model: str,
    ds_dict: datasets.DatasetDict,
    train_split: Sequence[int],
    eval_split: Sequence[int],
    #   class_names: Sequence[str] = CLASS_NAMES,
    #   train_ds: datasets.Dataset,
    #   eval_ds: datasets.Dataset,
    metarun_id: Union[int, str] = 0,
    run_id: Union[int, str] = 0,
    max_length: Optional[int] = 512,
    truncation: bool = True,
    return_tensors: str = "pt",
):
    ds_train_all = ds_dict["train"]
    class_names: Sequence[str] = ds_train_all.features["label"].names

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenized_ds_dict = ds_dict.map(
        lambda x: tokenizer(
            x["text"],
#            return_tensors=return_tensors,
            truncation=truncation,
            max_length=max_length,
        )
    )

    tokenized_ds_train_all = tokenized_ds_dict["train"]
    tokenized_train_ds = tokenized_ds_train_all.select(train_split)
    tokenized_eval_ds = tokenized_ds_train_all.select(eval_split)

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="max_length", max_length=max_length, return_tensors=return_tensors
    )

    id_label_enum = enumerate(class_names)
    id2label = {idx: label for idx, label in id_label_enum}
    label2id = {label: idx for idx, label in id_label_enum}

    num_labels=len(class_names)
    print(f"Number of labels: {num_labels}")
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels,
#        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
#        problem_type="multi_label_classification",
        #   problem_type="regression", # TODO (ordinal regression)
    )

    output_dir = f"../outputs/{lang}/{metarun_id}/{run_id}/"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split_info.out"), "w", encoding="utf-8") as f:
        print(f"Metarun {metarun_id}, Run {run_id} [all indices 1-indexed]", file=f)
        print(f"Train files: {[i+1 for i in train_split]}", file=f)
        print(f"Eval files: {[i+1 for i in train_split]}", file=f)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        eval_steps=1,
        evaluation_strategy="steps",
        log_level="debug",
        logging_steps=1,
        logging_strategy="steps",
        logging_first_step=True,
        save_steps=1,
        save_strategy="steps",
        #        save_strategy="epoch",
        seed=GLOBAL_SEED,
        load_best_model_at_end=True,
        resume_from_checkpoint=True,
        #        push_to_hub=True,
    )

    print(f"Training arguments: {training_args}")

    print(f"Tokenized train dataset: {tokenized_train_ds}")
    print(f"Tokenized eval dataset: {tokenized_eval_ds}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        #        train_dataset=ds_dict[train_split],
        #        eval_dataset=ds_dict[eval_split],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics(class_names),
    )

#    print(f"Trainer: {trainer}")
    print("Starting training...")
#    try:
#        trainer.train(resume_from_checkpoint=True)
#    except ValueError:
#        trainer.train()
    trainer.train()
    print("Training done!")


def main():
    lang = sys.argv[1]
    metarun_id = sys.argv[2] if len(sys.argv) >= 3 else "0"

    ds = assemble_dataset(lang, CLASS_LABELS)
    ds_all: datasets.Dataset = ds["train"]
    skfolds = StratifiedKFold(n_splits=N_FOLDS)
    #    skfolds = StratifiedKFold(n_splits=N_FOLDS, random_state=GLOBAL_SEED)

    for n, (train_idxs, eval_idxs) in enumerate(
        skfolds.split(range(ds_all.num_rows), ds_all["label"])
    ):
        print(f"#### FOLD {n} ####")
        print(f"Training entries: {train_idxs}")
        print(f"Validation entries: {eval_idxs}")
        finetune_for_sequence_classification(
            lang,
            PRETRAINED_MODEL,
            ds,
            train_split=train_idxs,
            eval_split=eval_idxs,
            metarun_id=metarun_id,
            run_id=f"split_{n}",
        )
        print(f"#### END FOLD {n} ####\n\n")


if __name__ == "__main__":
    main()
