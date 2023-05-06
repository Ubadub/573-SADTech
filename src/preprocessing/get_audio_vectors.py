"""
Create audio vectors.
"""

import argparse

from transformers import Wav2Vec2Processor, Wav2Vec2FeatureExtractor, PreTrainedTokenizer

import datasets


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Create vectors from the audio files in the given language's data directory.",
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

    feature_extractor = Wav2Vec2FeatureExtractor(ds_dict["train"]["audio"][0]["array"], return_tensors = "np")
    tokenizer = PreTrainedTokenizer()
    processor = Wav2Vec2Processor(feature_extractor, tokenizer)
    print(ds_dict["train"]["audio"][0])
    # print(feature_extractor."feature_size"[0]["array"])
    print(feature_extractor.feature_size)
    # processor.save_pretrained("test/")


if __name__ == "__main__":
    main()
