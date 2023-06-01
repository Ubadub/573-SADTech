"""
Creates audio vectors.

testing:
python3 pipeline_transformers/create_audio_vectors.py -d PATH/TO/DATASET -m MODEL

python3 -m pipeline_transformers  -d ../data/tam/train_dataset_dict train -c config/audio_vectorization/tfidf_wav2vec2_logistic.yml  -m ../outputs/debug/tam
"""

import argparse
import itertools
import logging
import os
from typing import Optional, Sequence, Union

import datasets

# from datasets import Audio, Dataset, DatasetDict, load_from_disk
# from datasets.features import Audio
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoFeatureExtractor,
    AutoModel,
    # AutoProcessor,
    MCTCTProcessor,
    PreTrainedModel,
    ProcessorMixin,
)

log = logging.getLogger(__name__)


class AudioFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Constructor

    Args:
        model_path:
            Defines the model to use for vectorization.
    """

    def __init__(
        self,
        model_path="Amrrs/wav2vec2-large-xlsr-53-tamil",
        layers_to_combine: Sequence[int] = [-1, -2, -3, -4],
        batch_size: Union[int, float] = 1.0,
        device: str = "cuda:0",
    ):
        self.model_path = model_path
        self.layers_to_combine = layers_to_combine
        self.device = device

        assert 0 < batch_size, "batch_size must be a positive float or int"

        self.batch_size = batch_size

        log.debug(
            f"Created AudioFeatureExtractor with:\n"
            f"\t\tmodel_path: {self.model_path}\n"
            f"\t\tlayers_to_combine: {self.layers_to_combine}\n"
            f"\t\tbatch_size: {self.batch_size}\n"
            f"\t\tdevice: {self.device}\n"
        )

    def _log_memory(self, header: str = None):
        if header:
            log.debug(header)
        log.debug(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
        log.debug(f"Cached: {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")

    def fit(self, X, y=None):
        log.debug("Fitting.")
        self.model_: PreTrainedModel = AutoModel.from_pretrained(self.model_path)
        self.feature_extractor_ = AutoFeatureExtractor.from_pretrained(self.model_path)
        # if "mctct" in self.model_.config.model_type:  # workaround
        #     log.debug("MCTCT model; using MCTCTProcessor instead of AutoProcessor.")
        #     self.processor_: ProcessorMixin = MCTCTProcessor.from_pretrained(
        #         self.model_path
        #     )
        # else:
        #     self.processor_: ProcessorMixin = AutoProcessor.from_pretrained(
        #         self.model_path
        #     )
        self._converter = datasets.Audio(
            sampling_rate=self.feature_extractor_.sampling_rate
            # sampling_rate=self.processor_.feature_extractor.sampling_rate
        )
        if getattr(self.model_.config, "is_encoder_decoder", False):
            log.debug("Model is encoder-decoder; saving encoder only.")
            self.model_ = self.model_.encoder

        return self

    def transform(self, X, y=None):
        """
        X should consist of a column of dictionaries with the key "bytes"
        """
        log.debug("Transforming.")
        self.model_.to(self.device)
        X = [self._converter.decode_example(x)["array"] for x in X]

        feats = self._get_features(audio_array=X)

        self.model_.to("cpu")  # free up GPU memory

        return feats

    @torch.no_grad()
    def _get_features(
        self,
        audio_array: Sequence[np.ndarray],
    ) -> np.ndarray:
        """
        Params:
            - audio_array: Sequence of array representations of audio files

        Creates and returns feature vectors for each audio array.
        """
        num_samples = len(audio_array)
        if type(self.batch_size) == float and self.batch_size <= 1.0:
            eff_batch_size = max(1, int(self.batch_size * num_samples))
        elif type(self.batch_size) == int:
            eff_batch_size = self.batch_size
        log.info(
            f"Beginning feature extraction using model {self.model_path}"
            f"with effective batch size {eff_batch_size}"
        )

        log.debug(f"Total samples: {num_samples}")

        inputs = self.feature_extractor_(
            # inputs = self.processor_(
            # audio_array[0],  # get the first one for testing purposes
            # batch,
            audio_array,
            sampling_rate=self._converter.sampling_rate,
            padding=True,
            return_tensors="pt",
        )  # We don't move this to GPU yet because the dataloader copies it

        inputs_keys, inputs_vals = zip(*inputs.items())
        inputs_tds = TensorDataset(*inputs_vals)
        loader = DataLoader(inputs_tds, batch_size=eff_batch_size)

        all_features = []

        if "cuda" in self.device:
            self._log_memory("Before all batches- Memory Usage:")

        for batch_idx, batch in enumerate(loader):
            log.info(f"Processing batch {batch_idx}.")
            if "cuda" in self.device:  # memory management
                torch.cuda.empty_cache()
            # log.info(f"Processing batch {batch_idx} of size {eff_batch_size}")
            log.info(f"Processing batch {batch_idx} of size {batch[0].shape[0]}")
            input_batch = dict(zip(inputs_keys, [_.to(self.device) for _ in batch]))

            if "cuda" in self.device:
                self._log_memory("Batch Start - Memory Usage:")

            outputs = self.model_(**input_batch, output_hidden_states=True)

            combined_layers = []
            # concatenating given layers
            for i in self.layers_to_combine:
                cur_state = outputs.hidden_states[i].detach().cpu().numpy()

                log.debug(f"State {i}:\n{cur_state}")
                log.debug(f"State {i} shape: {cur_state.shape}")

                averaged_cur_state = np.mean(cur_state, axis=1)

                combined_layers.append(averaged_cur_state)
            batch_features = np.concatenate(combined_layers, axis=1)
            log.debug(f"batch_features shape: {batch_features.shape}")
            all_features.append(batch_features)

            # Some memory management
            del outputs
            for tnsr in input_batch.values():
                tnsr.cpu()
            del input_batch

            log.debug(f"End batch {batch_idx}.")
            if "cuda" in self.device:
                self._log_memory("Batch End - Memory Usage:")

        concatenated_feats = np.concatenate(all_features, axis=0)

        log.debug(f"concatenated_feats shape: {concatenated_feats.shape}")
        log.info("Completed feature extraction.")

        return concatenated_feats


def main():
    logging.basicConfig()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:10240"

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create vectors from the audio files in the given language's dataset.",
    )
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument(
        "-c",
        "--layers_to_combine",
        nargs="*",
        default=[-1, -2, -3, -4],
        required=False,
        type=int,
    )
    parser.add_argument("-m", "--model_paths", nargs="+", required=True)
    parser.add_argument(
        "-l",
        "--logging_level",
        required=False,
        default="WARNING",
        choices=logging._nameToLevel.keys(),
    )
    parser.add_argument("-o", "--output", required=True)

    def int_or_float(s):
        if "." in s:
            v = float(s)
            if v > 0 and v <= 1:
                return v
        return int(s)

    parser.add_argument(
        "-b", "--batch_size", required=False, default=1.0, type=int_or_float
    )

    args = parser.parse_args()

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(args.dataset)

    log.setLevel(args.logging_level.upper())

    os.makedirs(args.output, exist_ok=True)

    extractor_kwargs = vars(args)
    extractor_kwargs = {
        k: v
        for k, v in extractor_kwargs.items()
        if k not in ("dataset", "logging_level", "model_paths", "output")
        and v is not None
    }

    new_ds_dict_builder: [str, datasets.Dataset] = {}

    for split in ds_dict:
        log.info(f"Processing split {split}.")
        ds: datasets.Dataset = ds_dict[split]
        ds_format = ds.format
        existing_format_columns = (
            ds_format.get("columns", []) if ds_format.get("type", "") == "numpy" else []
        )
        # ds: datasets.Dataset = ds_dict[split].select(range(2))  # for testing

        # audio_df: pd.Series = ds.to_pandas().iloc[0:2]["audio"] # alternate for testing
        audio_df: pd.Series = ds.to_pandas()["audio"]

        col_names = [
            f"{model_path}_{args.layers_to_combine}" for model_path in args.model_paths
        ]

        for idx, (model_path, col_name) in enumerate(zip(args.model_paths, col_names)):
            log.debug(f"Processing model {model_path}, using column name {col_name}")
            # extractor = AudioFeatureExtractor(**extractor_kwargs)
            audio_vectors: NDArray = (  # shape [dataset_length, hidden_dim]
                AudioFeatureExtractor(model_path=model_path, **extractor_kwargs)
                .fit(X=audio_df)
                .transform(X=audio_df)
            )

            log.debug(f"Vectors:\n{audio_vectors}")
            log.info(f"Created vector embedding with shape: {audio_vectors.shape}")

            ds = ds.add_column(name=col_name, column=list(audio_vectors))
            log.debug(f"Updated ds for split {split}:\n{ds}")
            log.debug(f"New features:\n{ds.features}")

            new_ds_dict_builder[split] = ds
            new_ds_dict: datasets.DatasetDict = datasets.DatasetDict(
                new_ds_dict_builder
            )
            columns_to_format = list(
                set(col_names[: idx + 1] + existing_format_columns)
            )
            new_ds_dict.set_format(
                type="numpy",
                columns=columns_to_format,
                output_all_columns=True,
            )
            log.debug(f"Reformatted columns {columns_to_format} to numpy.")

            new_ds_dict.save_to_disk(args.output)
            log.info(f"Saved dataset to {args.output}:\n{new_ds_dict}")

    # Model list
    # Malayalam:
    #     gvs/wav2vec2-large-xlsr-malayalam
    #     DrishtiSharma/whisper-large-v2-malayalam
    # Tamil:
    #     Amrrs/wav2vec2-large-xlsr-53-tamil
    #     vasista22/whisper-tamil-small
    #     speechbrain/m-ctc-t-large


if __name__ == "__main__":
    main()
