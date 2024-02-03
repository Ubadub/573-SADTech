from abc import abstractmethod, ABC
import copy
import logging
from typing import Any, Callable, Generic, Sequence, TypeVar

from datasets import load_from_disk, Dataset, DatasetDict
import fasttext
# from fasttext.FastText import _FastText as FastTextModel
import hydra
import nlpaug.augmenter.word as naw
from omegaconf import DictConfig, OmegaConf
import pandas as pd

log = logging.getLogger(__name__)

T = TypeVar("T")


class Augmentor(ABC, Generic[T]):
    """Dataset augmentor."""

    def __init__(self):
        """Constructor."""

    @abstractmethod
    def augment_example(self, example: T) -> Sequence[T]:
        """ """

    def augment_dataset(
        self,
        ds: Dataset,
        # augmentor: Callable,
        # path_to_model: str,
        column: str = "text",
        # ) -> Callable[[dict[str, Any]], dict[str, Any]]:
    ) -> Dataset:
        def _(batch: dict[str, Any]) -> dict[str, Any]:
            # ret_val = examples.copy()
            augmented_batch: dict[Any, list] = {k: [] for k in batch}
            # augmented = []
            # ret_val["column"] = augmented
            for original in batch[column]:
                augmented_samples = self.augment_example(original)
                augmented_batch[column].extend(augmented_samples)
                for k, v in batch.items():
                    if k != column:
                        augmented_batch[k].extend(
                            [copy.deepcopy(v) for _ in range(len(augmented_samples))]
                        )
            return augmented_batch

        return ds.map(
            function=_,
            batched=True,
            remove_columns=ds.column_names,
            features=ds.features,
        )


class FastTextAugmentor(Augmentor):
    def __init__(
        self,
        model_path: str,
        num_aug: int = 5,
        # min_sentence_length
    ):
        super().__init__()
        self._aug = naw.WordEmbsAug(model_type="fasttext", model_path=model_path, tokenizer=fasttext.tokenize)
        self.num_aug = num_aug
        # self.model = fasttext.load_model(path_to_model)

    def augment_example(self, example):
        aug_data = self._aug.augment(example, n=self.num_aug)
        log.debug(aug_data)
        return aug_data


# def fasttext_augment_dataset(
#     doc: str,
#     model: FastTextModel,
# ):
#     pass
#
#
# def augment_dataset(
#     ds: Dataset,
#     augmentor: Callable,
#     # path_to_model: str,
#     column: str = "text",
#     # ) -> Callable[[dict[str, Any]], dict[str, Any]]:
# ) -> Dataset:
#     def _(examples: dict[str, Any]) -> dict[str, Any]:
#         augmented = []
#         for original in examples["column"]:
#             augmented.append(augmentor(original))
#
#     return ds.map(function=_, batched=True, remove_columns=True)


@hydra.main(
    version_base=None, config_path="../config/augmentation", config_name="config"
)
def main(cfg: DictConfig) -> None:
    # log.debug(OmegaConf.to_yaml(cfg, resolve=False))
    log.debug(OmegaConf.to_yaml(cfg, resolve=True))
    # return
    ds_path: str = hydra.utils.to_absolute_path(cfg.dataset)
    splits_to_augment: Sequence[str] = cfg.splits_to_augment
    augment_target_split: str = cfg.augment_target_split
    ds_dict: DatasetDict = load_from_disk(ds_path)
    # identifier_col: str = cfg.identifier_col
    model_path: str = hydra.utils.to_absolute_path(cfg.model.path)
    save_path: str = hydra.utils.to_absolute_path(cfg.save_path)

    new_ds_dict_builder: dict = {}

    for split in splits_to_augment:
        log.info(
            f"Augmenting split {split}, placing augmented samples in split {augment_target_split}."
        )
        ds: Dataset = ds_dict[split]

        fta = FastTextAugmentor(model_path=model_path)
        ds_new = fta.augment_dataset(ds.select([1]))

        new_ds_dict_builder[augment_target_split] = ds_new

        # df: pd.DataFrame = ds.to_pandas()

    new_ds_dict: DatasetDict = DatasetDict(new_ds_dict_builder)
    new_ds_dict.save_to_disk(save_path)
    log.debug(f"Saved augmented dataset to {save_path}.")


if __name__ == "__main__":
    main()
