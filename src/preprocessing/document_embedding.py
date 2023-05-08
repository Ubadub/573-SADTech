"""
Document Embeddings: (TFIDF Weighted) Average Word Embeddings
"""

import sys
import yaml
import math
import os
import numpy as np

import datasets
import torch
import pickle
from transformers import AutoTokenizer, AutoModel


class TFIDF:

    def __init__(self, config: dict, data: datasets.Dataset) -> None:
        self.data = data
        self.output_dir = config["weights"]["tfidf_out_path"]
        self.lang = config["lang"]
        self.N = len(self.data)

        self.idf = {"<unk>": math.log(self.N / 2) + 1}
        self.tf = {}

        self.data.map(self.obtain_tf_idf_counts)
        self.obtain_idf()

        self.pickle_save()


    def obtain_tf_idf_counts(self, example: datasets.formatting.formatting.LazyRow) -> None:
        file = example["file"]
        tokens = example["tokens"]

        self.tf[(file, "<unk>")] = 1

        for token in tokens:
            file_token = (file, token)

            if token not in self.idf:
                self.idf[token] = 0
            if file_token not in self.tf:
                self.idf[token] += 1
            if file_token not in self.tf:
                self.tf[file_token] = 0
            self.tf[file_token] += 1


    def obtain_idf(self) -> None:
        for token, count in self.idf.items():
            idf = math.log(self.N / (1 + count)) + 1
            self.idf[token] = idf


    def __getitem__(self, item: tuple[str, str]) -> float:

        """
        file_name, token
        # TODO: take in index row mapped to a given document and a given token
        """
        if not isinstance(item, tuple):
            raise ValueError(f"Expected a tuple (file_name, token), got {item} instead")
        if len(item) != 2:
            raise ValueError(f"Expected a tuple (file_name, token), got {item} instead")

        file_name = item[0]
        token = item[1]

        if token not in self.idf:
            item = (file_name, "<unk>")
        elif item not in self.tf:
            item = (file_name, "<unk>")

        return self.tf[item] * self.idf[item[1]]


    def get_idf(self, token: str) -> float:
        if not isinstance(token, str):
            raise ValueError(f"Expected a token as a string, got {token} instead")

        if token not in self.idf:
            token = "<unk>"

        return self.idf[token]


    def get_tf(self, item: tuple[str, str]) -> float:
        if not isinstance(item, tuple):
            raise ValueError(f"Expected a tuple (file_name, token), got {item} instead")
        if len(item) != 2:
            raise ValueError(f"Expected a tuple (file_name, token), got {item} instead")

        if item not in self.tf:
            item[1] = "<unk>"

        return self.tf[item]


    def get_tfidf(self, item: tuple[str, str]) -> float:
        return self.__getitem__(item)


    def pickle_save(self):
        out_file = self.lang + "_tfidf.pickle"
        output_path = os.path.join(self.output_dir, out_file)

        with open(output_path, "wb") as out:
            pickle.dump(self, file=out, protocol=pickle.HIGHEST_PROTOCOL)


class DocumentEmbeddings:

    def __init__(self, config: dict, ds_dict: datasets.DatasetDict) -> None:
        """
            Params:
                - config: the dictionary read from a config.yml file
                - ds_dict: A HuggingFace Dataset Object with entries ["file", "label", "text"]

            This appends the entries ["token_indices", "tokens", "vectors"] to the given
                HuggingFace Dataset Object using the specified pretrained model in the config.yml file
                - "token_indices" list of token indices in a document after tokenization
                - "tokens" list of tokens in a document after tokenization
                - "vectors" binarization of numpy arrays that represent document embeddings created
                    by the (tfidf weighted) average of token vectors pulled from the specified pretrained model

                NOTE: token_indices and tokens have a one-to-one correspondence
        """

        self.config = config
        self.is_train_data = True if "train" in ds_dict else False
        self.data = ds_dict["train"] if "train" in ds_dict else ds_dict["test"]

        self.pretrained_model = config["pretrained_model"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.tokenize_dataset()

        self.model = AutoModel.from_pretrained(self.pretrained_model)
        self.vectorize()

        print(self.data)
        print(self.data["vectors"][:1][:10])


    def tokenizer_helper(self, example: datasets.formatting.formatting.LazyRow) -> dict[str, dict[str, list]]:
        """
        Params:
            - example: the text of a document

        Tokenizes the given text, creates a one-to=one mapping between token indices and tokens

        Returns:
            - A dictionary of dictionaries where "token_indices" is mapped to a dictionary containing
                the key "input_ids" mapped to token indices
        """
        example_len = len(example["text"])
        tokens = []
        window_size = 500
        for i in range(0, example_len, window_size):
            end = i + window_size if i + window_size < example_len else example_len
            cur_tokens = self.tokenizer(
                example["text"][i:end],
                truncation=False,
                return_attention_mask=False,
            )
            tokens.extend(cur_tokens["input_ids"])
        return {"token_indices": {"input_ids": tokens}}


    def tokenize_dataset(self) -> None:
        """
        Tokenizes the "text" entry of a HuggingFace dataset object and creates a new
            "tokens" entry storing the tokenized texts as a list of tokens
        """
        self.data = self.data.map(self.tokenizer_helper)

        self.data = self.data.map(
            lambda example: {"tokens": self.tokenizer.convert_ids_to_tokens(example["token_indices"]["input_ids"])}
        )


    # def fit(self, X, y=None) -> DocumentEmbeddings:
    #     return self



    # def transform(self, X: list[str], y=None) -> np.ndarray:
    # """
    #     Params:
    #         - X: matrix of shape [D, 1] where D is size of document set
    #         It is just the ['text'] column of the HuggingFace Dataset
    #
    #     Returns:
    #         - A matrix of shape [D, E] where D is the size of the document set and E is the size of
    #             the document embeddings (feature size)
    # """
    #
    # for i in range(len(X)):
    #     print(X[i])
    #
    # # TODO: iterate through the list of strings, call autotokenize and grab embeddings like usual





    @torch.no_grad()
    def vectorize(self):
        """
        This creates a new "vectors" entry to the given HuggingFace Dataset object. Stored
            in this dataset object is the binarization of a torch tensor. Each torch tensor
            represents the (weighted) average corresponding to the same "text" entry in the
            HuggingFace Dataset object.
        """
        word_embeddings = self.model.embeddings.word_embeddings.weight

        if self.is_train_data:
            if self.config["weights"]["tfidf_weighted"]:
                # get tfidf weighted average
                tfidf = TFIDF(config=self.config, data=self.data)
        else:
            pass
            #TODO: load in existing TFIDF created from train data

        self.data = self.data.map(
            self._vectorize_helper,
            fn_kwargs={
                "w_e": word_embeddings,
                "tfidf": tfidf
            }
        )


    @torch.no_grad()
    def _vectorize_helper(self,
                          example: datasets.formatting.formatting.LazyRow,
                          w_e: torch.nn.parameter.Parameter,
                          tfidf: TFIDF = None
                          ) -> None:
        """
            Params:
                - example: An entry of the HuggingFace Dataset object
                - w_e: The token embeddings obtained from the pretrained model's word embedding layer

            Takes an entry of a HuggingFace dataset, as well as the word embedding layer of the
                pretrained model. Creates a (weighted) average of the token vectors constituting the
                given entry of the HuggingFace dataset.
        """
        # grab the current indices mapped to the tokens that make up the current document
        input_ids = example["token_indices"]["input_ids"]

        print("w_e type:", type(w_e))

        # grab the current tensors corresponding to the tokens in the current document
        w_i = w_e[input_ids]

        if self.config["weights"]["tfidf_weighted"]:
            # get tfidf weighted average
            vectors = self._weighted_average(
                file_name=example["file"],
                word_embeddings=w_i,
                input_ids=input_ids,
                tfidf=tfidf
            )
        else:
            # get unweighted average
            vectors = torch.mean(w_i, dim=0)

        # transform torch tensor into numpy array
        # TODO: detatch to numpy first before doing any math
        vectors = vectors.detach().cpu().numpy()

        print(vectors[:5])

        vectors = pickle.dumps(vectors, protocol=pickle.HIGHEST_PROTOCOL)

        return {"vectors": vectors}


    def _weighted_average(self,
                          file_name: str,
                          word_embeddings: torch.Tensor,
                          input_ids: list,
                          tfidf: TFIDF) -> torch.Tensor:
        """
        Params:
            - file_name: The file
            - word_embeddings: A embeddings size [Document_size,
            - input_ids: A
            - tfidf:

        Returns:
            - A tfidf weighted average over all tokens in a document representing a document vector
        """
        print("word_embedding type:", type(word_embeddings))
        print("word_embeddings[0] type:", type(word_embeddings[0]))
        # print("word_embeddings[0][0] type:", type(word_embeddings[0]))
        num_tokens, embedding_size = word_embeddings.size()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        document_embedding = torch.zeros(size=(embedding_size,))

        for i in range(num_tokens):
            cur_embedding = word_embeddings[i]
            token = tokens[i]
            token_tfidf = tfidf[file_name, token]
            weighted_embedding = torch.mul(cur_embedding, token_tfidf)
            document_embedding = torch.add(document_embedding, weighted_embedding)

        print(f"Finished document embedding for {file_name}\n")

        return document_embedding


if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(config["data_path"])

    document_embeddings = DocumentEmbeddings(config, ds_dict)
