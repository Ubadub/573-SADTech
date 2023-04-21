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
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)

from config import CLASS_LABELS, CLASS_NAMES, GLOBAL_SEED, N_FOLDS

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

PRETRAINED_MODEL = "ai4bharat/indic-bert"


# FEATS = datasets.Features(
#     {"text": datasets.Value(dtype="string"), "label": CLASS_LABELS}
# )


def compute_metrics(class_names: Sequence[str]) -> Callable[[EvalPrediction], dict]:
    def _(eval_pred: EvalPrediction) -> dict:
        y_pred, y_true = eval_pred
        y_pred = np.argmax(y_pred, axis=1)
        metrics = classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
            output_dict=True,
        )
        print(classification_report(
            y_true,
            y_pred,
            labels=range(len(class_names)),
            target_names=class_names,
        ))
        return metrics

    return _


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
    inner_group_num: int = 2,
    truncation: bool = True,
    return_tensors: str = "pt",
    disable_tqdm: bool = False,
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
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
        return_tensors=return_tensors,
    )

    id_label_enum = enumerate(class_names)
    id2label = {idx: label for idx, label in id_label_enum}
    label2id = {label: idx for idx, label in id_label_enum}

    num_labels = len(class_names)
    print(f"Number of labels: {num_labels}")

    #    model_config = AutoConfig.from_pretrained(pretrained_model, num_hidden_groups=num_hidden_groups, num_labels=num_labels)
    print(f"Attempting to set inner group number to {inner_group_num}.")
    model_config = AutoConfig.from_pretrained(
        pretrained_model,
        inner_group_num=inner_group_num,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    print(f"Inner group number is now: {model_config.inner_group_num}.")

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        config=model_config,
        #   num_labels=num_labels,
        #   num_labels=len(class_names),
        #   problem_type="multi_label_classification",
        #   problem_type="regression", # TODO (ordinal regression)
    )

#    freeze_modules = [model.base_model.embeddings, model.base_model.encoder.albert_layer_groups[0]

    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False

    for param in model.base_model.encoder.albert_layer_groups[0].albert_layers[0].parameters():
        param.requires_grad = False

#    for idx, layer in enumerate(model.base_model.encoder.albert_layer_groups[0].albert_layers):
#        if idx
#        for param in model.base_model.encoder.albert_layer_groups[0].parameters():
#            param.requires_grad = False

#    if num_hidden_groups > 1:
#        pass

    output_dir = f"../outputs/{lang}/{metarun_id}/{run_id}/"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "split_info.out"), "w", encoding="utf-8") as f:
        print(f"Metarun {metarun_id}, Run {run_id} [all indices 1-indexed]", file=f)
        print(f"Train files: {[i+1 for i in train_split]}", file=f)
        print(f"Eval files: {[i+1 for i in train_split]}", file=f)

#    per_device_train_batch_size = min(tokenized_train_ds.num_rows, 16)
#        per_device_train_batch_size=,
#        per_device_eval_batch_size=tokenized_eval_ds.num_rows,

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
#        per_device_train_batch_size=tokenized_train_ds.num_rows,
#        per_device_eval_batch_size=tokenized_eval_ds.num_rows,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=100,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        log_level="debug",
        logging_strategy="epoch",
        logging_first_step=True,
        save_strategy="epoch",
#        evaluation_strategy="steps",
#        log_level="debug",
#        logging_steps=1,
#        logging_strategy="steps",
#        logging_first_step=True,
#        save_steps=1,
#        save_strategy="steps",
        #        save_strategy="epoch",
        seed=GLOBAL_SEED,
        load_best_model_at_end=True,
        resume_from_checkpoint=True,
        disable_tqdm=disable_tqdm,
        #        push_to_hub=True,
    )

#    print(f"Training arguments: {training_args}")
#
#    print(f"Tokenized train dataset: {tokenized_train_ds}")
#    print(f"Tokenized eval dataset: {tokenized_eval_ds}")

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

    trainer.save_model(os.path.join(output_dir, "best_model"))
    trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable. Continuing with CPU.")
        disable_tqdm = False
#        sys.exit("CUDA unavailable. Aborting.")
    else:
        print("CUDA available. Continuing.")
        datasets.disable_progress_bar()
        disable_tqdm = True

    lang = sys.argv[1]
    metarun_id = sys.argv[2] if len(sys.argv) >= 3 else "0"

    ds: datasets.DatasetDict = datasets.load_from_disk(
        f"../data/{lang}/train_dataset_dict"
    )
    ds_all: datasets.Dataset = ds["train"]
    skfolds = StratifiedKFold(n_splits=N_FOLDS)
    #    skfolds = StratifiedKFold(n_splits=N_FOLDS, random_state=GLOBAL_SEED)

    for n, (train_idxs, eval_idxs) in enumerate(
        skfolds.split(range(ds_all.num_rows), ds_all["label"])
    ):
        if n > 0:
            return
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
            disable_tqdm=disable_tqdm,
        )
        print(f"#### END FOLD {n} ####\n\n")


if __name__ == "__main__":
    main()
