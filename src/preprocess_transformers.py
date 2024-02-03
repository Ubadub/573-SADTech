from datasets import ClassLabel

from config import CLASS_LABELS, CLASS_NAMES, GLOBAL_SEED, N_FOLDS

def preprocess_twitter_xlm_roberta_base_sentiment(
    pretrained_model: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    save_path: str = "../outputs/models/xlm_roberta_twitter_sentiment_preprocessed",
):
    raw_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model)

    pass

def preprocess_indic_bert(
    labels: ClassLabel,
    pretrained_model: str = "ai4bharat/indic-bert",
    save_path: str = "../outputs/models/indic_bert_preprocessed",
    problem_type="regression", # "multi_label_classification"
    freeze_embeddings: bool = True,
    freeze_encoder: bool = True,
) -> PreTrainedModel:
    if problem_type == "multi_label_classification":
        num_labels = labels.num_classes
        id_label_enum = enumerate(labels.names)
        id2label = {idx: label for idx, label in id_label_enum}
        label2id = {label: idx for idx, label in id_label_enum}
    elif problem_type == "regression":
        num_labels = 1
        pass
    else:
        sys.exit("Unsupported problem type.")

    #   initial_model_config = AutoConfig.from_pretrained(pretrained_model)
    #   if "num_hidden_groups" in config_args:
    #       if initial_model_config["num_hidden_groups"] > config_args["num_hidden_groups"]:
    #           pass
    model_config = AutoConfig.from_pretrained(pretrained_model)
        pretrained_model,
        inner_group_num=inner_group_num,
        num_labels=num_labels,
        problem_type=problem_type,
        id2label=id2label,
        label2id=label2id,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_model,
        config=model_config,
        #   num_labels=num_labels,
        #   num_labels=len(class_names),
    )

    if freeze_embeddings:
        for param in model.base_model.embeddings.parameters():
            param.requires_grad = False

    if freeze_encoder:
        for param in model.base_model.encoder.albert_layer_groups[0].albert_layers[0].parameters():
            param.requires_grad = False

def main():
    preprocess_indic_bert(CLASS_LABELS)


if __name__ == "__main__":
    main()
