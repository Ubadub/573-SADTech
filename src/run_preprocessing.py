from config import CLASS_LABELS, CLASS_NAMES, GLOBAL_SEED, N_FOLDS

from preprocessing.dataset_creation import main as dataset_creation

if __name__ == "__main__":
    dataset_creation()