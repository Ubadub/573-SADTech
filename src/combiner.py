from datasets import DatasetDict, concatenate_datasets, load_from_disk

tam = load_from_disk("../data/tam/train_dataset_dict_audio")
mal = load_from_disk("../data/mal/train_dataset_dict_audio")

tam_train = tam["train"]
mal_train = mal["train"]

tam_mal = concatenate_datasets(
