from common import CLASS_LABELS, GLOBAL_SEED, N_FOLDS

from preprocessing.dataset_creation import main as dataset_creation
from preprocessing.get_audio_vectors import main as get_audio_vectors

if __name__ == "__main__":
    # dataset_creation()
    get_audio_vectors()
