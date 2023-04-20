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


    def train_predict(self):
        pass


    def kfold_validation(self, kfolds: int = N_FOLDS):
        skfolds = StratifiedKFold(n_splits=kfolds)  # does not shuffle
        splits = skfolds.split(X=self.feature_vectors, y=self.gold_labels)

        for n, (train_idxs, eval_idxs) in enumerate(tqdm(splits)):
            print(f"#### FOLD {n} ####")
            print(f"Training entries: {train_idxs}")
            print(f"Validation entries: {eval_idxs}")

            gold_labels, predicted = self.train_predict(train_idxs, eval_idxs)
            self.f1_score(gold_labels, predicted)

            print(f"#### END FOLD {n} ####\n\n")


    def f1_score(self, gold_labels, predicted):
        f1_weighted = f1_score(gold_labels, predicted, average="weighted")
        f1_macro = f1_score(gold_labels, predicted, average="macro")
        f1_micro = f1_score(gold_labels, predicted, average="micro")

        print("weighted f1 score:", f1_weighted)
        print("macro f1 score:", f1_macro)
        print("micro f1 score:", f1_micro)
