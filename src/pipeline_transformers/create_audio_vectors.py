"""
Creates audio vectors.

testing: python3 -m pipeline_transformers  -d ../data/tam/train_dataset_dict train -c config/audio_vectorization/tfidf_wav2vec2_logistic.yml  -m ../outputs/debug/tam
"""

from typing import Optional
import argparse
import numpy as np
import torch
import os
import json

import datasets
import transformers
from datasets.features import Audio

from sklearn.base import BaseEstimator, TransformerMixin
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    ClapFeatureExtractor,
    MCTCTFeatureExtractor,
)


UNK = "<@UNK!>"
PAD = "<PAD>"

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
            strategy: Optional[str] = "xlsr-wav2vec2",
            lang: Optional[str] = "tam"
    ):
        self.strategy = strategy
        self.lang = lang


    def fit(self, X, y=None, vocab_dir: str = "../outputs/vocab"):
        text = [example for example in X["text"]]

        vocab_dict = self._create_vocab_dict(text_data=text)

        self.vocab_file_path_ = os.path.join(vocab_dir, f"{self.lang}/vocab.json")

        with open(self.vocab_file_path_, "w") as vocab_out:
            json.dump(vocab_dict, vocab_out)

        return self

    def transform(self, X, y=None):
        # TODO: comment back in when done testing
        # converter = Audio(sampling_rate=16000)
        # X = [converter.decode_example(x)["array"] for x in X]
        X = [x["audio"]['array'] for x in X]

        if self.strategy == "wav2vec2":
            feature_extractor = self.get_wav2vec2_features(X)

        elif self.strategy == "xlsr-wav2vec2":
            return self._get_xlsr_wav2vec2_features(X)

        elif self.strategy == "clap":
            tokenizer = None
            feature_extractor = self.get_clap_features(X)
        elif self.strategy == "mctct":
            tokenizer = None
            feature_extractor = self.get_mctct_features(X)
        else:
            raise ValueError(f"Vectorization strategy should be wav2vec2, clap, xlsr-wav2vec2"
                             f" or mctct, but got {self.strategy} instead")


    def _create_vocab_dict(self, text_data: list[str]) -> dict[str, int]:
        vocab = self._extract_all_chars(text_data=text_data)

        vocab_dict = {char: index for index, char in enumerate(list(vocab))}

        # rename " " token to be more visible
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        # add unk token to vocabulary
        vocab_dict[UNK] = len(vocab_dict)
        # add <PAD> token to vocabulary
        vocab_dict[PAD] = len(vocab_dict)

        return vocab_dict

    @staticmethod
    def _extract_all_chars(text_data: list[str]) -> set:
        vocab = set()

        all_text = " ".join(text_data)
        for char in all_text:
            vocab.add(char)
        return vocab

    def _get_xlsr_wav2vec2_features(self, audio_array: list[np.ndarray]) -> np.ndarray:
        """
        Params:
            - audio_array: list of arrays representations of audio files

        Creates and returns XLSR-wav2vec2 feature vectors for each audio array.
        From: https://huggingface.co/docs/transformers/v4.29.1/en/model_doc/xlsr_wav2vec2
        """
        pretrained = ""

        # load in pretrained XLS-R Wav2Vec2 model
        model = transformers.Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-xls-r-300m"
            # "facebook/wav2vec2-xls-r-1b"
            # "facebook/wav2vec2-xls-r-2b"  # this is about 8.5GB

        )

        # create the Tokenizer
        # tokenizer = Wav2Vec2CTCTokenizer(
        #     vocab_file=self.vocab_file_path_,
        #     unk_token=UNK,
        #     pad_token=PAD,
        #     word_delimiter_token="|"
        # )

        # create the Feature Extractor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=True
        )

        # wrap the tokenizer and feature extractor in a Processor for convenience
        # processor = Wav2Vec2Processor(
        #     feature_extractor=feature_extractor,
        #     tokenizer=tokenizer
        # )

        inputs = processor(
            audio_array[0],  # get the first one for testing purposes
            # audio_array,
            sampling_rate=16000,
            padding=True,
            return_tensors="pt"
        )


        # grab inputs from the data
        inputs = feature_extractor(
            audio_array, return_tensors="pt", sampling_rate=16000, padding=True
        )


        print("inputs", inputs)

        print("HERE")

        # print(str(model))

        print("doing forward pass to retrieve embeddings")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # print("outputs", outputs)

        pooler_output = outputs.pooler_output
        print("pooler output", pooler_output)

        last_hidden_states = outputs.last_hidden_state

        print("vectors after forward", list(last_hidden_states.shape))

        asdf

        return inputs
    
    @staticmethod
    def get_wav2vec2_features(audio_array: list[np.ndarray]) -> np.ndarray:
        """
        Params:
            - audio_array: list of arrays representations of audio files

        Creates and returns wav2vec2 feature vectors for each audio array.
        From: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor
        """
        feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)
        inputs = feature_extractor(
            audio_array, return_tensors="np", sampling_rate=16000, padding=True
        )["input_values"]
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
        inputs = feature_extractor(
            audio_array, return_tensors="np", sampling_rate=16000, padding=True
        )["input_features"]
        return inputs

    @staticmethod
    def get_mctct_features(audio_array: list[np.ndarray]) -> np.ndarray:
        """
        Params:
            - audio_array: list of arrays representations of audio files

        Creates and returns M-CTC-T feature vectors for each audio array.
        From: https://huggingface.co/docs/transformers/model_doc/mctct#transformers.MCTCTFeatureExtractor
        """
        feature_extractor = MCTCTFeatureExtractor(sampling_rate=16000)
        inputs = feature_extractor(
            audio_array, return_tensors="np", sampling_rate=16000, padding=True
        )["input_features"]
        return inputs


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create vectors from the audio files in the given language's dataset.",
    )
    parser.add_argument("--lang")
    parser.add_argument("--strategy", default="wav2vec2")

    args = parser.parse_args()
    lang = args.lang

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(
        f"../data/{lang}/train_dataset_dict"
    )

    # audio_array = [audio_dict["array"] for audio_dict in ds_dict["train"]["audio"]]
    # print(audio_array)

    # vectors = AudioFeatureExtractor.get_wav2vec2_features(audio_array)
    # vectors = AudioFeatureExtractor.get_clap_features(audio_array)
    # vectors = AudioFeatureExtractor.get_mctct_features(audio_array)


    # audio_array = [audio_dict for audio_dict in ds_dict["train"]["audio"]]
    audio_vectors = AudioFeatureExtractor().fit(ds_dict["train"])
    vectors = audio_vectors.transform(X=ds_dict["train"])

    print(vectors)


if __name__ == "__main__":
    main()
