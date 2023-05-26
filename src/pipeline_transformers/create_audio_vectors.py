"""
Creates audio vectors.

testing: python3 -m pipeline_transformers  -d ../data/tam/train_dataset_dict train -c config/audio_vectorization/tfidf_wav2vec2_logistic.yml  -m ../outputs/debug/tam
"""

from typing import Optional, Sequence
import argparse
import numpy as np
import pandas as pd

import datasets
import torch
from datasets.features import Audio

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import (
    ProcessorMixin,
    Wav2Vec2Processor,
    Wav2Vec2Model,
    MCTCTProcessor,
    MCTCTModel,
    WhisperProcessor,
    WhisperModel
)


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

    def __init__(
            self,
            strategy: str = "wav2vec2",
            model: Optional[str] = "Amrrs/wav2vec2-large-xlsr-53-tamil",
            layers_to_combine: Sequence[int] = [-1, -2, -3, -4]
    ):
        self.strategy = strategy
        self.model = model
        self.layers_to_combine = layers_to_combine


    def fit(self, X, y=None):
        return self


    def transform(self, X, y=None):
        converter = Audio(sampling_rate=16000)
        X = [converter.decode_example(x)["array"] for x in X]

        if self.strategy == "wav2vec2":
            processor = Wav2Vec2Processor.from_pretrained(self.model)
            model = Wav2Vec2Model.from_pretrained(self.model)
        elif self.strategy == "whisper":
            processor = WhisperProcessor.from_pretrained(self.model)
            model = WhisperModel.from_pretrained(self.model)
        elif self.strategy == "mctct":
            processor = MCTCTProcessor.from_pretrained(self.model)
            model = MCTCTModel.from_pretrained(self.model)
        else:
            raise ValueError(f"Hello, pls pass in known vectorization strategy, either \
                             'Wav2Vec', 'mctct' or 'whisper'. Instead got {self.strategy}")

        return self.get_features(
            processor=processor,
            model=model,
            audio_array=X,
        )


    def get_features(self,
                     processor: ProcessorMixin,
                     model: torch.nn.Module,
                     audio_array: list[np.ndarray]) -> np.ndarray:
        """
        Params:
            - audio_array: list of arrays representations of audio files

        Creates and returns wav2vec2 feature vectors for each audio array.
        From: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor
        """
        inputs = processor(
            audio_array[0],  # get the first one for testing purposes
            # audio_array,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # concatenating given layers
        combined_layers = []
        for i in self.layers_to_combine:
            cur_state = outputs.hidden_states[i].detach().cpu().numpy()

            averaged_cur_state = np.mean(cur_state, axis=1)

            combined_layers.append(averaged_cur_state)

        return np.concatenate(combined_layers, axis=1)


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create vectors from the audio files in the given language's dataset.",
    )
    parser.add_argument("--lang")

    args = parser.parse_args()
    lang = args.lang

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(
        f"../data/{lang}/train_dataset_dict"
    )

    pd_audio = ds_dict["train"].to_pandas()["audio"]

    if lang == "mal":
        # to test Wav2Vec2 vectorizer
        # audio_vectors = AudioFeatureExtractor(model="gvs/wav2vec2-large-xlsr-malayalam").fit(X=pd_audio)

        # to test Whisper vectorizer
        # DrishtiSharma/whisper-large-v2-malayalam
        audio_vectors = AudioFeatureExtractor(model="DrishtiSharma/whisper-large-v2-malayalam").fit(X=pd_audio)
    else:
        # to test Wav2Vec2 vectorizer
        audio_vectors = AudioFeatureExtractor(model="Amrrs/wav2vec2-large-xlsr-53-tamil").fit(X=pd_audio)

        # to test Whisper vectorizer
        audio_vectors = AudioFeatureExtractor(model="vasista22/whisper-tamil-small").fit(X=pd_audio)

    # to test MCTCT vectorizer
    # audio_vectors = AudioFeatureExtractor(
    #     strategy="mctct",
    #     model="speechbrain/m-ctc-t-large"
    # ).fit(X=pd_audio)

    vectors = audio_vectors.transform(X=pd_audio)

    # print(vectors)
    print(vectors.shape)  # [batch_size, hidden_dim]


if __name__ == "__main__":
    main()
