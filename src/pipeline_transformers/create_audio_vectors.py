"""
Creates audio vectors.
"""

from typing import Optional
import argparse
import numpy as np
import pickle

import datasets

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import Wav2Vec2FeatureExtractor, ClapFeatureExtractor#, MCTCTFeatureExtractor

class AudioFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Constructor

    Args:
        strategy:
            Defines the vectorization strategy to do. Choose from:
            - wav2vec2 (Default)
            - clap
            - mctct
    """
    def __init__(self, strategy: Optional[str]):
        self.strategy = strategy

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print()
        print(type(X[0]))
        print(dict(X[0]).keys())
        print(pickle.loads(X[0]["bytes"]))
        print()
        audio_array = [audio_dict["array"] for audio_dict in X]

        if self.strategy == "wav2vec2":
            return self.get_wav2vec2_features(audio_array)
        elif self.strategy == "clap":
            return self.get_clap_features(audio_array)
        elif self.strategy == "mctct":
            return self.get_mctct_features(audio_array)
        else:
            return self.get_wav2vec2_features(audio_array)


    @staticmethod
    def get_wav2vec2_features(audio_array: list[np.ndarray]) -> np.ndarray:
        """
        Params:
            - audio_array: list of arrays representations of audio files

        Creates and returns wav2vec2 feature vectors for each audio array.
        From: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor
        """
        feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)
        inputs = feature_extractor(audio_array,
                                return_tensors="np",
                                sampling_rate=16000,
                                padding=True)["input_values"]
        return inputs


    @staticmethod
    def get_clap_features(audio_array: list[np.ndarray]) -> np.ndarray:
        """
        Params:
            - audio_array: list of arrays representations of audio files

        Creates and returns mel-filter bank feature vectors for each audio array.
        From: https://huggingface.co/docs/transformers/model_doc/clap#transformers.ClapFeatureExtractor
        """
        feature_extractor = ClapFeatureExtractor(sampling_rate=16000)
        inputs = feature_extractor(audio_array,
                                return_tensors="np",
                                sampling_rate=16000,
                                padding=True)["input_features"]
        return inputs


    # Needs torchaudio library
    @staticmethod
    def get_mctct_features(audio_array: list[np.ndarray]) -> np.ndarray:
        """
        Params:
            - audio_array: list of arrays representations of audio files

        Creates and returns M-CTC-T feature vectors for each audio array.
        From: https://huggingface.co/docs/transformers/model_doc/mctct#transformers.MCTCTFeatureExtractor
        """
        feature_extractor = MCTCTFeatureExtractor(sampling_rate=16000)
        inputs = feature_extractor(audio_array,
                                return_tensors="np",
                                sampling_rate=16000,
                                padding=True)["input_features"]
        return inputs


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create vectors from the audio files in the given language's dataset.",
    )
    parser.add_argument("lang")

    args = parser.parse_args()
    lang = args.lang

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(f"../data/{lang}/train_dataset_dict")

    # audio_array = [audio_dict["array"] for audio_dict in ds_dict["train"]["audio"]]
    # print(audio_array)

    # vectors = AudioFeatureExtractor.get_wav2vec2_features(audio_array)
    # vectors = get_clap_features(audio_array)
    # vectors = get_mctct_features(audio_array)

    # print(vectors)


if __name__ == "__main__":
    main()
