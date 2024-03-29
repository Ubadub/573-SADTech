import argparse
from collections import Counter
import logging
import os
import random
import sys
from typing import Any, Callable, Optional, Sequence, Union
import yaml

import datasets

# from datasets import Features, ClassLabel, Dataset, Value, load_dataset
from datasets import (
    ClassLabel,
    Dataset,
    DatasetDict,
    load_dataset,
    concatenate_datasets,
)
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
    DataCollator,
    DataCollatorWithPadding,
    EvalPrediction,
    PretrainedConfig,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)

from common import CLASS_LABELS, GLOBAL_SEED, N_FOLDS

# def configure_logger(log_level=LOG_LEVEL):
#     logger = logging.getLogger(__name__)
#     logging.basicConfig(
#         format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
#         datefmt="%m/%d/%Y %H:%M:%S",
#         handlers=[logging.StreamHandler(sys.stderr)],
#     )
#
#     logger.setLevel(log_level)
#     datasets.utils.logging.set_verbosity(log_level)
#     transformers.utils.logging.set_verbosity(log_level)
#     # set the main code and the modules it uses to the same log-level according to the node
#     log_level = training_args.get_process_log_level()


def compute_metrics(
    class_names: Sequence[str], regression: bool = False
) -> Callable[[EvalPrediction], dict]:
    def _(eval_pred: EvalPrediction) -> dict:
        y_pred, y_true = eval_pred
        num_classes = len(class_names)
        metrics = {"y_pred_raw": y_pred.squeeze().tolist()}
        if regression:
            # metrics["y_pred_raw"] = y_pred.squeeze().tolist()
            y_pred = np.fmin(
                num_classes - 1, np.fmax(0, y_pred.squeeze().round())
            )  # np.fmax(0, np.fmin(5, y_pred.squeeze().round()))
            # y_true = np.fmin(5, np.fmax(0, y_true.squeeze().round())) #np.fmax(0, np.fmin(5, y_true.squeeze().round()))
            metrics["MSE"] = mean_squared_error(y_true, y_pred)
            metrics["R^2"] = r2_score(y_true, y_pred)
        else:
            y_pred = np.argmax(y_pred, axis=1)

        metrics.update(
            classification_report(
                y_true,
                y_pred,
                labels=range(num_classes),
                target_names=class_names,
                output_dict=True,
            )
        )

        metrics["y_true"] = y_true.squeeze().tolist()
        metrics["y_pred"] = y_pred.squeeze().tolist()
        #        print(classification_report(
        #            y_true,
        #            y_pred,
        #            labels=range(len(class_names)),
        #            target_names=class_names,
        #        ))
        return metrics

    return _


# def assemble_model_from_pretrained(
#    pretrained_model: str,
#    model_config: PretrainedConfig,
# ) -> PretrainedModel:
#    pass
#
# def assemble_trainer(
#    model: PretrainedModel,
#    training_args: TrainingArguments,
#    tokenized_train_ds: Dataset,
#    tokenized_eval_ds: Dataset,
#    data_collator: DataCollator,
#    compute_metrics: Callable[[EvalPrediction], dict],
# ):
#    trainer = Trainer(
#        model=model,
#        args=training_args,
#        train_dataset=tokenized_train_ds,
#        eval_dataset=tokenized_eval_ds,
#        #        train_dataset=ds_dict[train_idxs],
#        #        eval_dataset=ds_dict[eval_idxs],
#        tokenizer=tokenizer,
#        data_collator=data_collator,
#        compute_metrics=compute_metrics,
#        #optimizers=
#        #compute_metrics=compute_regression_metrics, #compute_metrics(class_names),
#    )
#
# def tokenize(
#    tokenizer: PretrainedTokenizer,
#    ds: Union[Dataset, DatasetDict],
#    text_field: str = "text",
#    **tokenizer_kwargs,
# ) -> Union[Dataset, DatasetDict]:
##    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#    return ds_dict.map(
#        lambda x: tokenizer(
#            x[text_field],
#            **tokenizer_kwargs
#            #truncation=truncation,
#            #max_length=max_length,
#            #            return_tensors=return_tensors,
#        )
#    )


# def preprocess_indic_bert_for_regression(
#     pretrained_model: str,
#     inner_group_num: int = 2,
# ) -> PreTrainedModel:
#     model = preprocess_indic_bert
#     num_labels = 1 #len(class_names)
#     #   print(f"Number of labels: {num_labels}")
#
#     #    model_config = AutoConfig.from_pretrained(pretrained_model, num_hidden_groups=num_hidden_groups, num_labels=num_labels)
#     print(f"Attempting to set inner group number to {inner_group_num}.")
#     model_config = AutoConfig.from_pretrained(
#         pretrained_model,
#         inner_group_num=inner_group_num,
#         num_labels=num_labels,
#         problem_type="regression", # TODO (ordinal regression)
#         #   problem_type="multi_label_classification",
#         #   id2label=id2label,
#         #   label2id=label2id,
#     )
#     print(f"Inner group number is now: {model_config.inner_group_num}.")
#
#     model = AutoModelForSequenceClassification.from_pretrained(
#         pretrained_model,
#         config=model_config,
#         #   num_labels=num_labels,
#         #   num_labels=len(class_names),
#     )
#
# #    freeze_modules = [model.base_model.embeddings, model.base_model.encoder.albert_layer_groups[0]
#
#     for param in model.base_model.embeddings.parameters():
#         param.requires_grad = False
#
#     for param in model.base_model.encoder.albert_layer_groups[0].albert_layers[0].parameters():
#         param.requires_grad = False

# def old_finetune_for_sequence_classification(
#     lang: str,
#     pretrained_model: str,
#     ds_dict: DatasetDict,
#     training_args: TrainingArguments,
#     train_idxs: Sequence[int],
#     eval_idxs: Sequence[int],
#     class_names: Sequence[str] = CLASS_LABELS.names,
#     #   train_ds: Dataset,
#     #   eval_ds: Dataset,
#     max_length: Optional[int] = 512,
#     inner_group_num: int = 2,
#     truncation: bool = True,
#     return_tensors: str = "pt",
# ):
#     ds_train_all = ds_dict["train"]
#     #   class_names: Sequence[str] = ds_train_all.features["label"].names
#
#     tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#     tokenized_ds_dict = ds_dict.map(
#         lambda x: tokenizer(
#             x["text"],
#             #            return_tensors=return_tensors,
#             truncation=truncation,
#             max_length=max_length,
#         )
#     )
#
#     tokenized_ds_train_all = tokenized_ds_dict["train"]
#     tokenized_train_ds = tokenized_ds_train_all.select(train_idxs)
#     tokenized_eval_ds = tokenized_ds_train_all.select(eval_idxs)
#
#     data_collator = DataCollatorWithPadding(
#         tokenizer=tokenizer,
#         padding="max_length",
#         max_length=max_length,
#         return_tensors=return_tensors,
#     )
#
#     #   id_label_enum = enumerate(class_names)
#     #   id2label = {idx: label for idx, label in id_label_enum}
#     #   label2id = {label: idx for idx, label in id_label_enum}
#
#     num_labels = 1 #len(class_names)
#     #   print(f"Number of labels: {num_labels}")
#
#     #    model_config = AutoConfig.from_pretrained(pretrained_model, num_hidden_groups=num_hidden_groups, num_labels=num_labels)
#     print(f"Attempting to set inner group number to {inner_group_num}.")
#     model_config = AutoConfig.from_pretrained(
#         pretrained_model,
#         inner_group_num=inner_group_num,
#         num_labels=num_labels,
#         problem_type="regression", # TODO (ordinal regression)
#         #   problem_type="multi_label_classification",
#         #   id2label=id2label,
#         #   label2id=label2id,
#     )
#     print(f"Inner group number is now: {model_config.inner_group_num}.")
#
#     model = AutoModelForSequenceClassification.from_pretrained(
#         pretrained_model,
#         config=model_config,
#         #   num_labels=num_labels,
#         #   num_labels=len(class_names),
#     )
#
# #    freeze_modules = [model.base_model.embeddings, model.base_model.encoder.albert_layer_groups[0]
#
#     for param in model.base_model.embeddings.parameters():
#         param.requires_grad = False
#
#     for param in model.base_model.encoder.albert_layer_groups[0].albert_layers[0].parameters():
#         param.requires_grad = False
#
# #    for idx, layer in enumerate(model.base_model.encoder.albert_layer_groups[0].albert_layers):
# #        if idx
# #        for param in model.base_model.encoder.albert_layer_groups[0].parameters():
# #            param.requires_grad = False
#
# #    if num_hidden_groups > 1:
# #        pass
#
#     training_args = TrainingArguments(**training_args)
# #    per_device_train_batch_size = min(tokenized_train_ds.num_rows, 16)
# #        per_device_train_batch_size=,
# #        per_device_eval_batch_size=tokenized_eval_ds.num_rows,
#
# #    optimizer = torch.optim
# #    training_args = TrainingArguments(
# #        output_dir=output_dir,
# #        learning_rate=2e-5,
# ##        per_device_train_batch_size=tokenized_train_ds.num_rows,
# ##        per_device_eval_batch_size=tokenized_eval_ds.num_rows,
# #        per_device_train_batch_size=12,
# #        per_device_eval_batch_size=12,
# #        num_train_epochs=400,
# #        weight_decay=0.01,
# #        evaluation_strategy="epoch",
# #        log_level="debug",
# #        logging_strategy="epoch",
# #        logging_first_step=True,
# #        save_strategy="epoch",
# ##        evaluation_strategy="steps",
# ##        log_level="debug",
# ##        logging_steps=1,
# ##        logging_strategy="steps",
# ##        logging_first_step=True,
# ##        save_steps=1,
# ##        save_strategy="steps",
# #        #        save_strategy="epoch",
# #        seed=GLOBAL_SEED,
# #        load_best_model_at_end=True,
# #        resume_from_checkpoint=True,
# #        disable_tqdm=disable_tqdm,
# #        #        push_to_hub=True,
# #    )
#
# #    print(f"Training arguments: {training_args}")
# #
# #    print(f"Tokenized train dataset: {tokenized_train_ds}")
# #    print(f"Tokenized eval dataset: {tokenized_eval_ds}")
#
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=tokenized_train_ds,
#         eval_dataset=tokenized_eval_ds,
#         #        train_dataset=ds_dict[train_idxs],
#         #        eval_dataset=ds_dict[eval_idxs],
#         tokenizer=tokenizer,
#         data_collator=data_collator,
#         compute_metrics=compute_metrics(class_names, regression=True),
#         #optimizers=
#         #compute_metrics=compute_regression_metrics, #compute_metrics(class_names),
#     )
#
#     #    print(f"Trainer: {trainer}")
#     print("Starting training...")
#     #    try:
#     #        trainer.train(resume_from_checkpoint=True)
#     #    except ValueError:
#     #        trainer.train()
#
#     trainer.train()
#
#     print("Training done!")
#     final_model_path = os.path.join(output_dir, "final_model")
#     trainer.save_model(final_model_path )
#     trainer.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
#
#     return os.path.abspath(final_model_path)


def rebalance_ds(
    ds: Dataset,
    seed: Optional[int],
    label_field: str = "label",
    shuffle: bool = False,
    shuffle_seed: Optional[int] = None,
) -> Dataset:
    print("Beginning rebalancing.")
    if seed is not None:
        random.seed(seed)
    labels = ds[label_field]
    counts = Counter(labels)
    most_common_label, most_common_freq = counts.most_common()[0]
    new_ds = ds.filter(lambda x: x[label_field] == most_common_label)
    print(f"Most common label {most_common_label} occurs {most_common_freq } times.")
    for lbl in range(len(counts)):
        if lbl != most_common_label:
            print(f"Rebalancing for label {lbl} / {CLASS_LABELS.int2str(lbl)}")
            label_rows = ds.filter(lambda x: x[label_field] == lbl)
            lbl_cnt = len(label_rows)
            num_to_add = most_common_freq - lbl_cnt
            print(f"Adding {num_to_add} rows for {lbl} / {CLASS_LABELS.int2str(lbl)}")
            to_add_idxs = random.choices(range(lbl_cnt), k=num_to_add)
            print(f"Adding indices: {to_add_idxs}")
            to_add = label_rows.select(to_add_idxs)
            # label_rows = concatenate_datasets([label_rows, to_add])
            new_ds = concatenate_datasets([new_ds, label_rows, to_add])
    print(f"Dataset augmented from {len(ds)} entries to {len(new_ds)} entries long.")
    return new_ds.shuffle(seed=shuffle_seed) if shuffle else new_ds


def finetune_for_sequence_classification(
    lang: str,
    pretrained_model: str,
    ds_dict: DatasetDict,
    training_args_dict: dict,
    # training_args: TrainingArguments,
    train_idxs: Sequence[int],
    eval_idxs: Sequence[int],
    labels: ClassLabel,
    label_field: str = "label",
    #   class_names: Sequence[str] = CLASS_LABELS.names,
    #   train_ds: Dataset,
    #   eval_ds: Dataset,
    max_length: Optional[int] = 512,  # 514, #512,
    num_layers_to_freeze: int = 0,
    rebalance: bool = True,
    regression: bool = False,
    truncation: bool = True,
):
    if regression:
        print("RUNNING REGRESSION.")
        model_config = AutoConfig.from_pretrained(pretrained_model, num_labels=1)
        ds_dict = ds_dict.cast_column(label_field, datasets.Value("float64"))
    else:
        print("RUNNING MULTICLASS CLASSIFICATION.")
        id_label_enum = enumerate(labels.names)
        id2label = {idx: label for idx, label in id_label_enum}
        label2id = {label: idx for idx, label in id_label_enum}
        model_config = AutoConfig.from_pretrained(
            pretrained_model,
            num_labels=labels.num_classes,
            id2label=id2label,
            label2id=label2id,
        )

    #    raw_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model, config=model_config, ignore_mismatched_sizes=True
    )

    print("Freezing embeddings.")
    for param in model.base_model.embeddings.parameters():
        param.requires_grad = False

    if model.config.model_type == "albert":
        print(
            f"Setting hidden mapping in requires_grad to: {num_layers_to_freeze <= 0}"
        )
        for param in model.base_model.encoder.embedding_hidden_mapping_in.parameters():
            param.requires_grad = num_layers_to_freeze <= 0

        for layer_group_idx, layer_group in enumerate(
            model.base_model.encoder.albert_layer_groups
        ):
            for param in layer_group.parameters():
                param.requires_grad = layer_group_idx >= num_layers_to_freeze
            if layer_group_idx >= num_layers_to_freeze:
                print(f"Layer group {layer_group_idx} NOT frozen.")
            else:
                print(f"Layer group {layer_group_idx} frozen.")

    else:  # BERT - TODO: make this an actual check
        print(f"Freezing {num_layers_to_freeze} layers.")
        for layer_idx, layer in enumerate(model.base_model.encoder.layer):
            for param in layer.parameters():
                param.requires_grad = layer_idx >= num_layers_to_freeze
            if layer_idx >= num_layers_to_freeze:
                print(f"Layer {layer_idx} NOT frozen.")
            else:
                print(f"Layer {layer_idx} frozen.")

    #    with torch.no_grad():
    #        model_state_dict = model.state_dict()
    #        model_state_dict[

    ds_train_all = ds_dict["train"]
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    tokenized_ds_dict = ds_dict.map(
        lambda x: tokenizer(
            x["text"],
            truncation=truncation,
            max_length=max_length,
        )
    )

    tokenized_ds_train_all = tokenized_ds_dict["train"]
    tokenized_train_ds = tokenized_ds_train_all.select(train_idxs)
    tokenized_eval_ds = tokenized_ds_train_all.select(eval_idxs)

    if rebalance:
        # seed = training_args_dict.get("seed")
        # tokenized_train_ds = rebalance_ds(tokenized_train_ds, seed=seed, shuffle=True, shuffle_seed=seed)
        tokenized_train_ds = rebalance_ds(
            tokenized_train_ds,
            seed=training_args_dict.get("seed"),
            shuffle=True,
            shuffle_seed=training_args_dict.get("seed"),
        )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )

    training_args: TrainingArguments = TrainingArguments(**training_args_dict)
    print(
        f"Training per_device_train_batch_size: {training_args.per_device_train_batch_size}."
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        #        train_dataset=ds_dict[train_idxs],
        #        eval_dataset=ds_dict[eval_idxs],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics(labels.names, regression=regression),
        # optimizers=
        # compute_metrics=compute_regression_metrics, #compute_metrics(labels.names),
    )

    #    print(f"Trainer: {trainer}")
    print("Starting training...")

    trainer.train()

    print("Training done! Saving trainer state and best model...")

    final_model_path = os.path.abspath(
        os.path.join(training_args.output_dir, "final_model")
    )
    trainer.save_model(final_model_path)
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )

    print(f"Saved final model to {final_model_path}.")

    return final_model_path


def train(config_path: dict):
    # with open(args.config, "r") as f:
    with open(config_path, "r") as f:
        config = yaml.unsafe_load(f.read())

    base_pretrained_model = config["pretrained_model"]
    lang = config["lang"]
    run_id = config["run_id"]
    label_field = config.get("label_field", "label")
    dataset_dict_path = config.get(
        "dataset_dict_path", f"../data/{lang}/train_dataset_dict"
    )
    train_split = config.get("train_split", "train")
    n_splits = config.get("n_splits", 4)
    config["TrainingArguments"]["no_cuda"] = config["TrainingArguments"].get(
        "no_cuda", False
    )
    use_gpu = config.get("use_gpu", not config["TrainingArguments"]["no_cuda"])
    config["TrainingArguments"]["output_dir"] = config["TrainingArguments"].get(
        "output_dir", f"../outputs/{lang}"
    )
    disable_loading_bar = config.get(
        "disable_loading_bar", config["TrainingArguments"].get("disable_tqdm", False)
    )
    config["TrainingArguments"]["disable_tqdm"] = config["TrainingArguments"].get(
        "disable_tqdm", disable_loading_bar
    )
    #    freeze_protocol = config.get("FreezingProtocol", [0])
    base_training_args: dict = config.get("TrainingArguments", {})
    base_finetuning_args: dict = config.get("FinetuningArguments", {})
    phases = config["phases"]

    if torch.cuda.is_available():
        if use_gpu:
            print("CUDA available. Continuing.")
        else:
            print("CUDA available, but refusing to use.")
            config["TrainingArguments"]["no_cuda"] = True
    #            if config["TrainingArguments"].get("no_cuda"):
    #                print("Overriding TrainingArguments.no_cuda due to config use_gpu setting and refusing to use GPU.")
    #            else:
    #                config["TrainingArguments"]["no_cuda"] = True
    else:
        if use_gpu:
            sys.exit("CUDA unavailable, but config asked for it. Aborting!")
        else:
            config["TrainingArguments"]["no_cuda"] = True
            print("CUDA unavailable. Continuing with CPU.")

    if disable_loading_bar:
        print("DISABLING LOADING BAR")
        datasets.disable_progress_bar()

    ds: datasets.DatasetDict = datasets.load_from_disk(dataset_dict_path)
    ds_all: datasets.Dataset = ds[train_split]

    # ds = ds.cast_column(label_field, datasets.Value("float64"))

    skfolds = StratifiedKFold(n_splits)
    #   skfolds = StratifiedKFold(n_splits=N_FOLDS)
    #    skfolds = StratifiedKFold(n_splits=N_FOLDS, random_state=GLOBAL_SEED)

    #    print(config)
    #    return
    for n, (train_idxs, eval_idxs) in enumerate(
        skfolds.split(range(ds_all.num_rows), ds_all[label_field])
    ):
        #        if n > 0:
        #            return
        print(f"#### BEGIN FOLD {n} ####")
        print(f"Training entries: {train_idxs}")
        print(f"Validation entries: {eval_idxs}")

        curr_pretrained_model = base_pretrained_model
        #        training_args = base_training_args.copy()
        #        finetuning_args = base_finetuning_args.copy()
        #        print("finetuning_args:", finetuning_args)
        #        for num_layers_to_freeze in freeze_protocol:
        for phase_idx, phase in enumerate(phases):
            training_args = base_training_args.copy()
            finetuning_args = base_finetuning_args.copy()
            print(f"#### BEGIN TRAINING PHASE {phase_idx} ####\n\n")
            print("Phase training arguments are:", phase["TrainingArguments"])

            training_args.update(phase["TrainingArguments"])
            finetuning_args.update(phase.get("FinetuningArguments", {}))

            print(f"Training args for phase {phase_idx}:", training_args)

            training_args["output_dir"] = os.path.join(
                base_training_args["output_dir"], f"{run_id}/fold_{n}/phase{phase_idx}"
            )
            os.makedirs(training_args["output_dir"], exist_ok=True)
            with open(
                os.path.join(training_args["output_dir"], "split_info.out"),
                "w",
                encoding="utf-8",
            ) as f:
                print(f"Run {run_id}, phase {phase_idx}, fold {n} ", file=f)
                print(
                    f"Train files [all indices 1-indexed]: {[i+1 for i in train_idxs]}",
                    file=f,
                )
                print(
                    f"Eval files [all indices 1-indexed]: {[i+1 for i in eval_idxs]}",
                    file=f,
                )

            #        training_args = TrainingArguments(**config["TrainingArguments")
            curr_pretrained_model = finetune_for_sequence_classification(
                lang=lang,
                pretrained_model=curr_pretrained_model,
                ds_dict=ds,
                training_args_dict=training_args,
                train_idxs=train_idxs,
                eval_idxs=eval_idxs,
                labels=CLASS_LABELS,
                # num_layers_to_freeze=num_layers_to_freeze,
                **finetuning_args,
            )
            print(f"#### END TRAINING PHASE {phase_idx} ####\n\n")
        curr_pretrained_model = base_pretrained_model
        print(f"#### END FOLD {n} ####\n\n")


#    parser.add_argument("-l," "--lang", required=True)
#    parser.add_argument("-i", "--run_id", required=True, default="run0")
#    parser.add_argument("-c", "--model_config", required=True)
#    parser.add_argument("-t", "--trainer_config", required=True)
#    parser.add_argument("-o", "--output_dir")

#    parser.add_argument("-d", "--data_dir", default="../data/")
#    parser.add_argument("-l", "--labels_file", default="all.csv")
#    parser.add_argument("-m", "--delimiter", default=",")
#    config = args.config
#    model_config = args.model_config
#    trainer_config = args.trainer_config
