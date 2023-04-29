from argparse import ArgumentParser
import sys

from .transformer_inference import infer


def train(*args, **kwargs):
    raise NotImplementedError


PARSER_CONFIG = {
    "prog": "python -m transformers",
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
parser.add_argument("lang")
subparsers = parser.add_subparsers(**SUBPARSERS_CONFIG)

inf_subparser = subparsers.add_parser(
    "infer", aliases=["i", "inf"], description="Do inference"
)
inf_subparser.set_defaults(func=infer)
train_subparser = subparsers.add_parser(
    "train", aliases=["t", "tr", "finetune"], description="Finetune"
)
train_subparser.set_defaults(func=train)

args = parser.parse_args()

func_kwargs = dict(vars(args))
del func_kwargs["func"]

# sys.exit()

args.func(**func_kwargs)
