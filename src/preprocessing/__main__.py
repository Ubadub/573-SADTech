import argparse
import os

from datasets import DatasetDict

from common import CLASS_LABELS
from .dataset_creation import assemble_dataset

parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description="Generate a `datasets.DatasetDict` from the text files in a given language's data directory and save it to file.",
)
parser.add_argument("lang")
parser.add_argument(
    "-o", "--output", help="Directory to store the processed dataset in."
)
parser.add_argument(
    "-d",
    "--drop_file",
    action="store_true",
    help="Include this argument if the filename of each row in the dataset should be dropped as a column.",
)
parser.add_argument(
    "-t",
    "--tokenizer",
    help="The name of a pretrained tokenizer to use; if provided, the output of the tokenizer will be included amongst the output dataset's columns.",
)
# parser.add_argument(
#     "-m",
#     "--model",
#     help="The name (or path) of a pretrained model to use to generate vectors for each element of the dataset; if provided, the output of the tokenizer will be included amongst the output dataset's columns.",
# )
#    parser.add_argument("-d", "--data_dir", default="../data/")
#    parser.add_argument("-l", "--labels_file", default="all.csv")
#    parser.add_argument("-m", "--delimiter", default=",")

args = parser.parse_args()
output_path = args.output or f"../data/{args.lang}/train_dataset_dict"
output_dir_path = os.path.dirname(output_path)

if output_dir_path:
    os.makedirs(output_dir_path, exist_ok=True)

ds_dict: DatasetDict = assemble_dataset(
    args.lang,
    class_labels=CLASS_LABELS,
    drop_file=args.drop_file,
    tokenizer=args.tokenizer,
    model=args.model,
)
ds_dict.save_to_disk(output_path)
