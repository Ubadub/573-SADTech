from abc import abstractmethod, ABC
import pickle

from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from common import CLASS_LABELS, GLOBAL_SEED, N_FOLDS
from preprocessing.create_vectors import Vectors


class Classifier(ABC):
    """
    Wrapper around a `sklearn.base.BaseEstimator` for training models with a given
    vectorization scheme over a given `datasets.Dataset` instance.
    """

    CFG_MODEL_FIELD = "model"
    CFG_MODEL_ARGS_FIELD = "ModelArguments"
    CFG_MODEL_KWARGS_FIELD = "ModelKeywordArguments"
    CFG_VECTORIZATION_ARGS_FIELD = "VectorizationArguments"
    CFG_VECTORIZATION_KWARGS_FIELD = "VectorizationKeywordArguments"

    def __init__(self, config: dict):
        """
        Constructor.

        Args:
            config:
                A dictionary (e.g. created from a YAML config file) containing various
                configuration settings and variables.
                Should contain (key : value):
                    `Classifer.CFG_MODEL`:
                        a class that subclasses from `sklearn.base.BaseEstimator`.

                Optionally may contain:
                    `Classifer.CFG_MODEL_ARGS`:
                        any keyword arguments to pass to the model constructor.

                    `Classifer.CFG_MODEL_KWARGS`:
                        any keyword arguments to pass to the model constructor.
        """
        self._cfg = config
        model_args: Iterable = self._cfg.get(self.CFG_MODEL_ARGS_FIELD, []) or []
        model_kwargs: dict = self._cfg.get(self.CFG_MODEL_KWARGS_FIELD, {}) or {}
        self._vectorization_args: Iterable = (
            self._cfg.get(self.CFG_VECTORIZATION_ARGS_FIELD, []) or []
        )
        self._vectorization_kwargs: dict = (
            self._cfg.get(self.CFG_VECTORIZATION_KWARGS_FIELD, {}) or {}
        )
        if self.CFG_MODEL_FIELD in self._cfg:
            if issubclass(self._cfg[self.CFG_MODEL_FIELD], BaseEstimator):
                self._model: BaseEstimator = self._cfg[self.CFG_MODEL_FIELD](
                    *model_args, **model_kwargs
                )
            else:
                raise ValueError(
                    f"Field {self.CFG_MODEL_FIELD} in config file must be a subclass of "
                    "sklearn.base.BaseEstimator, but is actually of type "
                    f"{type(self._cfg[self.CFG_MODEL_FIELD])}."
                )
        else:
            raise ValueError(f"Config file must have {self.CFG_MODEL_FIELD} field.")
        # self._ds = ds  # ds_dict["train"]
        # feature_vector_wrapper = Vectors(config=config, ds_dict=ds_dict)
        # self.feature_vectors = feature_vector_wrapper.get_vectors().toarray()
        # self.gold_labels = np.array(ds_dict["train"]["label"])

    @abstractmethod
    def _vectorize(
        self, ds: Dataset, *args, text_field: str = "text", **kwargs
    ) -> np.ndarray:
        """Vectorize the given dataset. The vectorization algorithm/protocol to use
        should be implemented by subclasses of this class.

        Given a dataset of num_instances entries, returns an `np.ndarray` instance of
        shape (num_instances, num_features) where num_features is determined by the
        vectorization algorithm, which should be implemented by subclasses of this
        class.

        Args:
            ds:
                The dataset to use, containing num_instances rows.
            text_field:
                name of the field in ds that holds the text to vectorize.

        Returns:
            An `np.ndarray` instance of shape (num_instances, num_features) where
            num_features is determined by the vectorization algorithm.
        """

    def train(self, train_ds: Dataset, label_field: str = "label") -> None:
        """Train the model using the given training dataset.

        Args:
            ds:
                The dataset to use, containing num_instances rows.
            label_field:
                name of the field in ds that holds the class label.
        """
        X = self._vectorize(
            train_ds, *self._vectorization_args, **self._vectorization_kwargs
        )
        print("Shape of X:", X.shape)
        y_true = train_ds[label_field]
        self._model = self._model.fit(X, y_true)

    def predict(self, eval_ds: Dataset) -> np.ndarray:
        """Perform classification on the given dataset.

        Args:
            eval_ds:
                The dataset to use, containing num_instances rows.

        Returns:
            An `np.ndarray` instance of shape (num_instances,) where each entry contains
            a corresponding class prediction.
        """
        X = self._vectorize(
            eval_ds, *self._vectorization_args, **self._vectorization_kwargs
        )
        return self._model.predict(X)

    def save_to_file(self, fpath: str) -> None:
        """Save an instance of this class via pickling to a given file.

        Args:
            fpath:
                Path to the file to which the pickled data will be written.
        """
        with open(fpath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def from_file(cls, fpath: str) -> "Classifier":
        """Load an instance of this class from the pickled data in a file.

        Args:
            fpath:
                Path to the file containing the pickled data.

        Returns:
            An instance of this class, unpickled from the given file.
        """
        with open(fpath, "rb") as f:
            return pickle.load(f)


class OldClassifier:
    def kfold_validation(self, kfolds: int = N_FOLDS) -> None:
        """
        Params:
            - kfolds: number of folds to be run for kfolds validation

        Performs kfolds validation over a training set. Outputs results and
            and evaluation metrics.
        """
        skfolds = StratifiedKFold(n_splits=kfolds)  # does not shuffle
        splits = skfolds.split(X=self.feature_vectors, y=self.gold_labels)

        with open(self._cfg["results_path"] + "/D2_scores.out", "a") as output:
            output.write("#### Lang: " + self._cfg["lang"] + " ####\n")
            if self._cfg["classifier"] == "nb":
                output.write("#### Naive Bayes Evaluation Output ####\n")
            else:
                raise ValueError(
                    "Please include possible classifier in config.yml file."
                )

        prec_avg = []
        acc_avg = []
        f1_avg = []

        for n, (train_idxs, eval_idxs) in enumerate(splits):
            # print(f"#### FOLD {n} ####")
            # print(f"Training entries: {train_idxs}")
            # print(f"Validation entries: {eval_idxs}")

            gold_labels, predicted = self.train_predict(train_idxs, eval_idxs)

            self.output_predicted_labels(gold_labels, predicted, eval_idxs, n)

            self.output_f1_score(gold_labels, predicted, n, prec_avg, acc_avg, f1_avg)

            # print(f"#### END FOLD {n} ####\n\n")

        prec_avg = sum(prec_avg) / len(prec_avg)
        acc_avg = sum(acc_avg) / len(acc_avg)
        f1_avg = sum(f1_avg) / len(f1_avg)

        with open(self._cfg["results_path"] + "/D2_scores.out", "a") as output:
            output.write(f"#### Pooled F1 Scores ####\n")
            output.write("weighted average precision score: " + str(prec_avg) + "\n")
            output.write("accuracy score: " + str(acc_avg) + "\n")
            output.write("weighted average f1 score: " + str(f1_avg) + "\n")
            output.write(f"#### END ####\n\n")

    def output_predicted_labels(
        self,
        gold_labels: np.ndarray,
        predicted: np.ndarray,
        eval_idxs: np.ndarray,
        fold_num: int,
    ) -> None:
        """
        Params:
            - gold_labels: list of gold labels
            - predicted: list of predicted labels
            - eval_idxs: list of evaluation indexes
            - fold_num: fold number

        Outputs the file name, gold label, and predicted label for the given gold labels
        and predicted labels to the output path specified by the config.yml file.
        """
        with open(self._cfg["output_path"] + "/nb_output.txt", "a") as output:
            output.write(f"#### FOLD {fold_num} ####\n")
            output.write("file name \t gold label \t predicted label\n")
            for idx, file_idx in enumerate(eval_idxs):
                idx = int(idx)
                output.write(
                    self._ds[int(file_idx)]["file"]
                    + "\t"
                    + CLASS_LABELS.names[int(gold_labels[idx])]
                    + "\t\t"
                    + CLASS_LABELS.names[int(predicted[idx])]
                    + "\n"
                )
            output.write(f"#### END FOLD {fold_num} ####\n\n")

    def output_f1_score(
        self,
        gold_labels: np.ndarray,
        predicted: np.ndarray,
        fold_num: int,
        prec_avg: list[float],
        acc_avg: list[float],
        f1_avg: list[float],
    ) -> None:
        """
        Param:
            - gold_labels: np.ndarray of gold labels with same indexing scheme as argument predicted
            - predicted: np.ndarray of predicted labels, with same indexing scheme as argument
                         gold_labels
            - prec_avg: list of precision metric scores
            - acc_avg: list of accuracy metric scores
            - f1_avg: list of f1 metric scores

        Prints the precision, accuracy, and f1 scores to the results path specified by the
            config.yml file.

        # TODO: make a metrics class and print out more helpful information/output to chosen file??
        """

        scores = classification_report(
            gold_labels, predicted, output_dict=True, zero_division=0
        )

        prec = scores["weighted avg"]["precision"]
        acc = scores["accuracy"]
        f1 = scores["weighted avg"]["f1-score"]

        prec_avg.append(prec)
        acc_avg.append(acc)
        f1_avg.append(f1)

        with open(self._cfg["results_path"] + "/D2_scores.out", "a") as output:
            output.write(f"#### FOLD {fold_num} ####\n")
            output.write("weighted average precision score: " + str(prec) + "\n")
            output.write("accuracy score: " + str(acc) + "\n")
            output.write("weighted average f1 score: " + str(f1) + "\n")
            output.write(f"#### END FOLD {fold_num} ####\n\n")
