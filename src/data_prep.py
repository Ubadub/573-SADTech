import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tam_stop_words import STOP_WORDS as TAM_STOP_WORDS
from mal_stop_words import STOP_WORDS as MAL_STOP_WORDS


def clean_up_line(line: str, stop_words: set) -> list:
    """
    things removed:
        punctuation: . , ; \\ \\" \\' ?
        numbers: 0-9
    """
    line = re.sub(r"\.\s|,|;|\\\\|\\\"|\\'|\?|\d", "", line)
    line = " ".join([word for word in line.split()
                        if word not in stop_words])
    return line.strip()


def get_file_words(directory: str, stop_words: set) -> dict[str, list]:
    file_words = {}
    for file in os.listdir(directory):
        file_path = directory + file
        file_name = file.split(".")[0]
        file_words[file_name] = ""
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                line = clean_up_line(line, stop_words)
                file_words[file_name] += " " + line
            file_words[file_name] = file_words[file_name].strip()
    return file_words


def vectorize(X_train, X_val):
    """
    from https://skimai.com/fine-tuning-bert-for-sentiment-analysis/
    """
    # Preprocess text
    X_train_preprocessed = np.array([text for text in X_train])
    X_val_preprocessed = np.array([text for text in X_val])

    # Calculate TF-IDF
    tf_idf = TfidfVectorizer(ngram_range=(1, 3),
                            binary=True,
                            smooth_idf=False)
    X_train_tfidf = tf_idf.fit_transform(X_train_preprocessed)
    X_val_tfidf = tf_idf.transform(X_val_preprocessed)

    return X_train_tfidf, X_val_tfidf


def get_vectors(lang, stop_words):
    values = get_file_words("data/" + lang + "/text/", stop_words)
    values = sorted(values.items())
    values = [x[1] for x in values]

    labels = pd.read_csv("data/" + lang + "/" + lang + "_train_label.tsv", sep="\t", header=0)
    labels = list(labels["Label"])

    values = values[0:len(labels)]

    X_train, X_val, y_train, y_val =\
    train_test_split(values, labels, test_size=0.2, random_state=2020)

    return vectorize(X_train, X_val)


def main():
    tam_X_train_tfidf, tam_X_val_tfidf = get_vectors("tam", TAM_STOP_WORDS)
    np.savez("outputs/vectors/tam_X_train_tfidf.npz", data=tam_X_train_tfidf.data, indices=tam_X_train_tfidf.indices,
             indptr=tam_X_train_tfidf.indptr, shape=tam_X_train_tfidf.shape)
    np.savez("outputs/vectors/tam_X_val_tfidf.npz", data=tam_X_val_tfidf.data, indices=tam_X_val_tfidf.indices,
             indptr=tam_X_val_tfidf.indptr, shape=tam_X_val_tfidf.shape)


    mal_X_train_tfidf, mal_X_val_tfidf = get_vectors("mal", MAL_STOP_WORDS)
    np.savez("outputs/vectors/mal_X_train_tfidf.npz", data=mal_X_train_tfidf.data, indices=mal_X_train_tfidf.indices,
             indptr=mal_X_train_tfidf.indptr, shape=mal_X_train_tfidf.shape)
    np.savez("outputs/vectors/mal_X_val_tfidf.npz", data=mal_X_val_tfidf.data, indices=mal_X_val_tfidf.indices,
             indptr=mal_X_val_tfidf.indptr, shape=mal_X_val_tfidf.shape)


if __name__ == '__main__':
    main()