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
from datasets.features import Audio
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import (
    AutoModel,
    AutoProcessor,
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

        # if type(batch_size) == float:
        #     assert 0 < batch_size and batch_size <= 1.0, "Float batch size must be in range (0, 1]"

        self.batch_size = batch_size

    def _log_memory(self, header: str = None):
        if header:
            log.debug(header)
        log.debug(f"Allocated: {round(torch.cuda.memory_allocated(0)/1024**3, 1)} GB")
        log.debug(f"Cached: {round(torch.cuda.memory_reserved(0)/1024**3, 1)} GB")

    def fit(self, X, y=None):
        self.model_: PreTrainedModel = AutoModel.from_pretrained(self.model_path)
        self.processor_: ProcessorMixin = AutoProcessor.from_pretrained(self.model_path)
        self._converter = Audio(
            sampling_rate=self.processor_.feature_extractor.sampling_rate
        )
        return self

    def transform(self, X, y=None):
        """
        X should consist of a column of dictionaries with the key "bytes"
        """
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

        Creates and returns wav2vec2 feature vectors for each audio array.
        From: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor
        """
        num_samples = len(audio_array)
        if type(self.batch_size) == float and self.batch_size <= 1.0:
            eff_batch_size = max(
                1, int(self.batch_size * num_samples)
            )  # max(1, int(1.0/self.batch_size))
        elif type(self.batch_size) == int:
            eff_batch_size = self.batch_size
        log.info(
            f"Beginning feature extraction using model {self.model_path} with effective batch size {eff_batch_size}"
        )

        log.debug(f"Total samples: {num_samples}")
        # batch_starts = range(0, num_samples, eff_batch_size)
        # batch_ends = itertools.chain(batch_starts[1:], [num_samples])

        # combined_layers = []
        # for batch_start, batch_end in np.array_split(range(num_samples), max(int(num_samples/eff_batch_size), 1))
        # log.debug(f"Processing batch from {batch_start} to {batch_end-1}")

        # batch = audio_array[batch_start:batch_end]
        inputs = self.processor_(
            # audio_array[0],  # get the first one for testing purposes
            # batch,
            audio_array,
            sampling_rate=self._converter.sampling_rate,
            padding=True,
            return_tensors="pt",
        )  # .to(self.device)

        inputs_keys, inputs_vals = zip(*inputs.items())
        # collate_fn = lambda batch: {keys[i]: batch[i] for i in range(len(batch))}

        inputs_tds = TensorDataset(*inputs_vals)
        loader = DataLoader(inputs_tds, batch_size=eff_batch_size)

        all_features = []

        if "cuda" in self.device:
            self._log_memory("Before all batches- Memory Usage:")

        for batch_idx, batch in enumerate(loader):
            log.info(f"Processing batch {batch_idx}.")
            if "cuda" in self.device:
                torch.cuda.empty_cache()
            # log.info(f"Processing batch {batch_idx} of size {len(batch)}")
            # for tnsr in batch:
            #     tnsr.to(self.device)
            input_batch = dict(zip(inputs_keys, [_.to(self.device) for _ in batch]))

            # for k, v in input_batch.items():
            #     log.debug(f"Tensor type: {type(v)}")

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
    parser.add_argument("-m", "--model_path", required=False)
    parser.add_argument(
        "-l",
        "--logging_level",
        required=False,
        default="WARNING",
        choices=logging._nameToLevel.keys(),
    )

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
    model_path: str = args.model_path

    # logging.basicConfig(level=args.logging_level.upper())
    # log = logging.getLogger(__name__)

    log.setLevel(args.logging_level.upper())
    extractor_kwargs = vars(args)
    # extractor_kwargs.pop("dataset")
    extractor_kwargs = {
        k: v
        for k, v in extractor_kwargs.items()
        if k not in ("dataset", "logging_level") and v is not None
    }

    audio_df: pd.Series = ds_dict["train"].to_pandas()["audio"]

    audio_vectors = (
        AudioFeatureExtractor(**extractor_kwargs)
        .fit(X=audio_df)
        .transform(X=audio_df)
        # AudioFeatureExtractor(**({"model_path": model_path} if args.model_path else {}))
    )
    # if lang == "mal":
    #     # to test Wav2Vec2 vectorizer
    #     # audio_vectors = AudioFeatureExtractor(model="gvs/wav2vec2-large-xlsr-malayalam").fit(X=audio_df)

    #     # to test Whisper vectorizer
    #     # DrishtiSharma/whisper-large-v2-malayalam
    #     audio_vectors = AudioFeatureExtractor(
    #         model="DrishtiSharma/whisper-large-v2-malayalam"
    #     ).fit(X=audio_df)
    # else:
    #     # to test Wav2Vec2 vectorizer
    #     audio_vectors = AudioFeatureExtractor(
    #         model="Amrrs/wav2vec2-large-xlsr-53-tamil"
    #     ).fit(X=audio_df)

    #     # to test Whisper vectorizer
    #     audio_vectors = AudioFeatureExtractor(
    #         model="vasista22/whisper-tamil-small"
    #     ).fit(X=audio_df)

    # to test MCTCT vectorizer
    # audio_vectors = AudioFeatureExtractor(
    #     strategy="mctct",
    #     model="speechbrain/m-ctc-t-large"
    # ).fit(X=audio_df)

    # vectors = audio_vectors.transform(X=audio_df)

    log.debug(f"Vectors:\n{audio_vectors}")
    log.debug(f"Vector shape: {audio_vectors.shape}")  # [batch_size, hidden_dim]


if __name__ == "__main__":
    main()
