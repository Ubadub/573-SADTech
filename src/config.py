import datasets

GLOBAL_SEED = 573

N_FOLDS = 4

CLASS_NAMES = (
    "HIGHLY NEGATIVE",
    "NEGATIVE",
    "NEUTRAL",
    "POSITIVE",
    "HIGHLY POSITIVE",
)

CLASS_LABELS = datasets.ClassLabel(names=CLASS_NAMES)
