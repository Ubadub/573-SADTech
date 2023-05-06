"""
"""

import os

import librosa
from pydub import AudioSegment

import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor, Wav2Vec2FeatureExtractor

import datasets


def convert_to_wav(directory):
    """
    Params:
        - directory: path to directory containing audio files

    Converts all mp3 files in directory to wav and saves them in the directory.
    """
    audio_files = os.listdir(directory)
    for file in audio_files:
        audio = AudioSegment.from_mp3(directory + "/" + file)
        audio.export(directory + "/" + file.split(".")[0] + ".wav", format="wav")


def get_batch_sequences(directory):
    batch_sequences = {}

    audio_files = os.listdir(directory)
    for file in audio_files:
        if file.endswith(".wav"):
            audio, rate = librosa.load(directory + "/" + file, sr = 16000)
            batch_sequences[file.split(".")[0]] = audio

    return batch_sequences

def main():
    # Convert mal and tam mp3 files to wav
    # convert_to_wav("data/mal/audio")
    # convert_to_wav("data/tam/audio")

    # mal_audio = get_batch_sequences("data/mal/audio")
    # tam_audio = get_batch_sequences("data/tam/audio")

    mal_audio: datasets.DatasetDict = datasets.load_from_disk("../data/mal/train_dataset_dict")
    tam_audio: datasets.DatasetDict = datasets.load_from_disk("../data/mal/train_dataset_dict")

    mal_vectors = Wav2Vec2FeatureExtractor(ds_dict["train"]["audio"], return_tensors = "np")
    tam_vectors = Wav2Vec2FeatureExtractor(ds_dict["train"]["audio"], return_tensors = "np")

    # with open("test.txt", "w") as wf:
    #     wf.write(str(mal_vectors.__dict__))


if __name__ == "__main__":
    main()
