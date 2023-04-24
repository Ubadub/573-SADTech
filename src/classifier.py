import datasets
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from config import CLASS_LABELS, CLASS_NAMES, GLOBAL_SEED, N_FOLDS
from preprocessing.create_vectors import Vectors


class Classifier:

    def __init__(self, config: dict, ds_train: datasets.DatasetDict):
        self.config = config
        feature_vector_wrapper = Vectors(config=config, ds_dict=ds_train)
        self.feature_vectors = feature_vector_wrapper.get_vectors().toarray()
        self.gold_labels = np.array(ds_train["train"]["label"])
        self.model = None
        self.dataset = ds_train["train"]


    def train_predict(self, train_idxs: np.array, eval_idxs: np.array) -> tuple[np.array, np.array]:
        """
        Abstract Method, necessary for any classifier

        This class is supposed to implement both the training and inference time of a model, returning
            gold labels and predicted labels (in that order) in a tuple
        """
        raise NotImplementedError("Please implement the train_predict() method to override this one!")


    def kfold_validation(self, kfolds: int = N_FOLDS) -> None:
        """
            Params:
                - kfolds: number of folds to be run for kfolds validation

            Performs kfolds validation over a training set. Outputs results and
                and evaluation metrics.
        """
        skfolds = StratifiedKFold(n_splits=kfolds)  # does not shuffle
        splits = skfolds.split(X=self.feature_vectors, y=self.gold_labels)

        with open(self.config["results_path"] + "/D2_scores.out", "a") as output:
            output.write("#### Lang: " + self.config["lang"] + " ####\n")
            if self.config["classifier"] == "nb":
                output.write("#### Naive Bayes Evaluation Output ####\n")
            else:
                raise ValueError("Please include possible classifier in config.yml file.")

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

        with open(self.config["results_path"] + "/D2_scores.out", "a") as output:
            output.write(f"#### Pooled F1 Scores ####\n")
            output.write("weighted average precision score: " + str(prec_avg) + "\n")
            output.write("accuracy score: " + str(acc_avg) + "\n")
            output.write("weighted average f1 score: " + str(f1_avg)+ "\n")
            output.write(f"#### END ####\n\n")


    def output_predicted_labels(self, gold_labels: np.array, predicted: np.array,
                                eval_idxs: np.array, fold_num: int) -> None:
        """
            Params:
                - gold_labels: list of gold labels
                - predicted: list of predicted labels
                - eval_idxs: list of evaluation indexes
                - fold_num: fold number

            Outputs the file name, gold label, and predicted label for the given gold labels and
                predicted labels to the output path specified by the config.yml file.
        """
        with open(self.config["output_path"] + "/nb_output.txt", "a") as output:
            output.write(f"#### FOLD {fold_num} ####\n")
            output.write("file name \t gold label \t predicted label\n")
            for idx, file_idx in enumerate(eval_idxs):
                idx = int(idx)
                output.write(self.dataset[int(file_idx)]["file"] + "\t" + CLASS_NAMES[int(gold_labels[idx])] + \
                      "\t\t" + CLASS_NAMES[int(predicted[idx])] + "\n")
            output.write(f"#### END FOLD {fold_num} ####\n\n")


    def output_f1_score(self, gold_labels: np.array, predicted: np.array, fold_num: int,
                        prec_avg: list[float], acc_avg: list[float],
                        f1_avg: list[float]) -> None:
        """
        Param:
            - gold_labels: np.array of gold labels with same indexing scheme as argument predicted
            - predicted: np.array of predicted labels, with same indexing scheme as argument
                         gold_labels
            - prec_avg: list of precision metric scores
            - acc_avg: list of accuracy metric scores
            - f1_avg: list of f1 metric scores

        Prints the precision, accuracy, and f1 scores to the results path specified by the
            config.yml file.

        # TODO: make a metrics class and print out more helpful information/output to chosen file??
        """

        scores = classification_report(gold_labels, predicted, output_dict=True)

        prec = scores["weighted avg"]["precision"]
        acc = scores["accuracy"]
        f1 = scores["weighted avg"]["f1-score"]

        prec_avg.append(prec)
        acc_avg.append(acc)
        f1_avg.append(f1)

        with open(self.config["results_path"] + "/D2_scores.out", "a") as output:
            output.write(f"#### FOLD {fold_num} ####\n")
            output.write("weighted average precision score: " + str(prec) + "\n")
            output.write("accuracy score: " + str(acc) + "\n")
            output.write("weighted average f1 score: " + str(f1)+ "\n")
            output.write(f"#### END FOLD {fold_num} ####\n\n")
