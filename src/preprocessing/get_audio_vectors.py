"""
Creates audio vectors.
"""

import argparse
import numpy as np

import datasets

from transformers import Wav2Vec2FeatureExtractor, ClapFeatureExtractor, MCTCTFeatureExtractor


def get_wav2vec2_features(audio_array: list[np.ndarray], file_names: list[str]) -> dict[str, np.ndarray]:
    """
    Params:
        - audio_array: list of arrays representations of audio files
        - file_names: list of file names in the current dataset

    Creates and returns wav2vec2 feature vectors for each audio array.
    From: https://huggingface.co/docs/transformers/v4.28.1/en/model_doc/wav2vec2#transformers.Wav2Vec2FeatureExtractor

    Returns:
        - A dictionary that maps each audio file name to its feature vector.
    """
    feature_extractor = Wav2Vec2FeatureExtractor(sampling_rate=16000)
    inputs = feature_extractor(audio_array,
                               return_tensors="np",
                               sampling_rate=16000,
                               padding=True)["input_values"]

    return get_vector_dict(inputs, file_names)

def get_clap_features(audio_array: list[np.ndarray], file_names: list[str]) -> dict[str, np.ndarray]:
    """
    Params:
        - audio_array: list of arrays representations of audio files
        - file_names: list of file names in the current dataset

    Creates and returns mel-filter bank feature vectors for each audio array.
    From: https://huggingface.co/docs/transformers/model_doc/clap#transformers.ClapFeatureExtractor

    Returns:
        - A dictionary that maps each audio file name to its feature vector.
    """
    feature_extractor = ClapFeatureExtractor(sampling_rate=16000)
    inputs = feature_extractor(audio_array,
                               return_tensors="np",
                               sampling_rate=16000,
                               padding=True)["input_features"]

    return get_vector_dict(inputs, file_names)


# Needs torchaudio library
def get_mctct_features(audio_array: list[np.ndarray], file_names: list[str]) -> dict[str, np.ndarray]:
    """
    Params:
        - audio_array: list of arrays representations of audio files
        - file_names: list of file names in the current dataset

    Creates and returns M-CTC-T feature vectors for each audio array.
    From: https://huggingface.co/docs/transformers/model_doc/mctct#transformers.MCTCTFeatureExtractor

    Returns:
        - A dictionary that maps each audio file name to its feature vector.
    """
    feature_extractor = MCTCTFeatureExtractor(sampling_rate=16000)
    inputs = feature_extractor(audio_array,
                               return_tensors="np",
                               sampling_rate=16000,
                               padding=True)["input_features"]

    return get_vector_dict(inputs, file_names)


def get_vector_dict(inputs, file_names):
    vectors = {}
    for file_name, input in zip(file_names, inputs):
        vectors[file_name] = input
    return vectors


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create vectors from the audio files in the given language's dataset.",
    )
    parser.add_argument("lang")
    parser.add_argument("-o", "--output")
    #    parser.add_argument("-d", "--data_dir", default="../data/")
    #    parser.add_argument("-l", "--labels_file", default="all.csv")
    #    parser.add_argument("-m", "--delimiter", default=",")

    args = parser.parse_args()
    lang = args.lang
    # output_path = args.output or f"../data/{lang}/train_dataset_dict"
    # output_dir_path = os.path.dirname(output_path)

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(f"../data/{lang}/train_dataset_dict")

    audio_array = [audio_dict["array"] for audio_dict in ds_dict["train"]["audio"]]
    file_names = ds_dict["train"]["file"]

    # vectors = get_wav2vec2_features(audio_array, file_names)
    # vectors = get_clap_features(audio_array, file_names)
    # vectors = get_mctct_features(audio_array, file_names)

    # print(type(vectors))


if __name__ == "__main__":
    main()
