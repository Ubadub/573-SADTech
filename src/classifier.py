import datasets
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from config import CLASS_LABELS, CLASS_NAMES, GLOBAL_SEED, N_FOLDS
from preprocessing.create_vectors import Vectors


class Classifier:

    def __init__(self, config: dict, ds_train: datasets.DatasetDict):
        self.config = config
        feature_vector_wrapper = Vectors(config=config, ds_dict=ds_train)
        self.feature_vectors = feature_vector_wrapper.get_vectors().toarray()
        self.gold_labels = np.array(ds_train["train"]["label"])
        self.model = None


    def train_predict(self, train_idxs: np.array, eval_idxs: np.array) -> tuple[np.array, np.array]:
        """
        Abstract Method, necessary for any classifier

        This class is supposed to implement both the training and inference time of a model, returning
            gold labels and predicted labels (in that order) in a tuple
        """
        raise NotImplementedError("Please implement the train_predict() method to override this one!")


    def kfold_validation(self, kfolds: int = N_FOLDS) -> None:
        skfolds = StratifiedKFold(n_splits=kfolds)  # does not shuffle
        splits = skfolds.split(X=self.feature_vectors, y=self.gold_labels)

        for n, (train_idxs, eval_idxs) in enumerate(tqdm(splits)):
            print(f"#### FOLD {n} ####")
            print(f"Training entries: {train_idxs}")
            print(f"Validation entries: {eval_idxs}")

            gold_labels, predicted = self.train_predict(train_idxs, eval_idxs)
            #TODO: output predicted labels to a file(?) to inspect choices our model is making
            self.f1_score(gold_labels, predicted)

            print(f"#### END FOLD {n} ####\n\n")


    def f1_score(self, gold_labels: np.array, predicted: np.array) -> None:
        """
        Param:
            - gold_labels: np.array of gold labels with same indexing scheme as argument predicted
            - predicted: np.array of predicted labels, with same indexing scheme as argument gold_labels

        Prints the weighted, macro, and micro f1 scores to the console
        # TODO: make a metrics class and print out more helpful information/output to chosen file??
        """

        f1_weighted = f1_score(gold_labels, predicted, average="weighted")
        f1_macro = f1_score(gold_labels, predicted, average="macro")
        f1_micro = f1_score(gold_labels, predicted, average="micro")

        print("weighted f1 score:", f1_weighted)
        print("macro f1 score:", f1_macro)
        print("micro f1 score:", f1_micro)
