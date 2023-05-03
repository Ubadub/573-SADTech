from argparse import ArgumentParser
import sys

from .finetune_transformer import train
from .transformer_inference import infer


PARSER_CONFIG = {
    "prog": "python -m transformer_lm",
    "description": "Finetune or do inference with transformer LMs.",
}

SUBPARSERS_CONFIG = {
    "title": "Action",
    "description": "Action to execute- finetuning or inference.",
    # "dest": "action",
    "required": True,
    "help": "Select finetune to do finetuning or infer to do inference.",
    # "metavar": f"[{', '.join(filters.CLI_FILTERS.keys())}]",
}

parser = ArgumentParser(**PARSER_CONFIG)
subparsers = parser.add_subparsers(**SUBPARSERS_CONFIG)

inf_subparser = subparsers.add_parser(
    "infer", aliases=["i", "inf"], description="Do inference"
)
inf_subparser.set_defaults(func=infer)
inf_subparser.add_argument(
    "-l",
    "--lang",
    required=True,
    choices=["tam", "mal"],
    help="Language (tam for Tamil, mal for Malayalam)",
)
train_subparser = subparsers.add_parser(
    "train",
    aliases=["t", "tr", "finetune"],
    description="Finetune a pretrained model from HuggingFace",
)
train_subparser.add_argument(
    "-c",
    "--config",
    required=True,
    dest="config_path",
    metavar="PATH_TO_CONFIG.YML",
    help="Path to the config YAML file for this train run.",
)
train_subparser.set_defaults(func=train)

args = parser.parse_args()

func_kwargs = dict(vars(args))
del func_kwargs["func"]

# sys.exit()

args.func(**func_kwargs)
