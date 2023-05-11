"""
Document Embeddings: (Optional) TFIDF Weighted Average Word Embeddings
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
from sklearn.base import BaseEstimator, TransformerMixin



class TFIDF:

    TFIDF_UNK = "<@tfidf_unk$>"

    def __init__(
            self,
            data: list[str],

    ) -> None:
        self.data = data
        self.N = len(self.data)

        self.idf = {self.TFIDF_UNK: 0}
        self.tf = {}

        # calculate tfidf scores
        self.obtain_tf_and_idf_counts()
        self.obtain_idf()

        # self.pickle_save()


    def obtain_tf_and_idf_counts(self) -> None:

        example_num = 0
        for example in self.data:

            self.tf[(example_num, self.TFIDF_UNK)] = 1

            for token in example:
                file_token = (example_num, token)

                if token not in self.idf:
                    self.idf[token] = 0

                # document not counted yet in document freq
                if file_token not in self.tf:
                    self.idf[token] += 1
                if file_token not in self.tf:
                    self.tf[file_token] = 0
                self.tf[file_token] += 1

            example_num += 1


    def obtain_idf(self) -> None:
        for token, count in self.idf.items():
            idf = math.log(self.N / (1 + count))
            self.idf[token] = idf


    def __getitem__(self, item: tuple[int, str]) -> float:
        """
            Params:
                -file_num: An int representation of the current document
                -token: A token as a string

            Returns:
                - the tfidf score of the given token in the given document as a float
        """
        if not isinstance(item, tuple):
            raise ValueError(f"Expected a tuple (file_name, token), got {item} instead")
        if len(item) != 2:
            raise ValueError(f"Expected a tuple (file_name, token), got {item} instead")

        file_name = item[0]
        token = item[1]

        if (token not in self.idf) or (item not in self.tf):
            item = (file_name, self.TFIDF_UNK)

        return self.tf[item] * self.idf[item[1]]


    def get_idf(self, token: str) -> float:
        """
            Params:
                -token: A token as a string

            Returns:
                - the idf score of the given token as a float
        """
        if not isinstance(token, str):
            raise ValueError(f"Expected a token as a string, got {token} instead")

        if token not in self.idf:
            token = self.TFIDF_UNK

        return self.idf[token]


    def get_tf(self, item: tuple[str, str]) -> int:
        """
            Params:
                - item: A tuple containing
                    -file_num: An int representation of the current document
                    -token: A token as a string
                Note: Must be a single tuple, unlike self.__getitem__()

            Returns:
                - the tf score of the given token in the given document as an int
        """
        if not isinstance(item, tuple):
            raise ValueError(f"Expected a tuple (file_name, token), got {item} instead")
        if len(item) != 2:
            raise ValueError(f"Expected a tuple (file_name, token), got {item} instead")

        if item not in self.tf:
            item = (item[0], self.TFIDF_UNK)

        return self.tf[item]


    def get_tfidf(self, item: tuple[str, str]) -> float:
        """
            Params:
                -file_num: An int representation of the current document
                -token: A token as a string

        Same as calling tfidf[] or self.tfidf.__get__item()

            Returns:
                - the tfidf score of the given token in the given document as a float
        """
        return self.__getitem__(item)


    # def pickle_save(self):
    #     out_file = self.lang + "_tfidf.pickle"
    #     output_path = os.path.join(self.output_dir, out_file)
    #
    #     with open(output_path, "wb") as out:
    #         pickle.dump(self, file=out, protocol=pickle.HIGHEST_PROTOCOL)


class DocumentEmbeddings(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            language_model: str,
            tfidf_output_path: str,
            tfidf_weighted: bool = True,
    ) -> None:
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
        # with open(config_path, "r") as ymlfile:
        #     self.config = yaml.load(ymlfile, Loader=yaml.Loader)
        # self.config = self.config["TextTransformers"][0]["kwargs"]
        super().__init__()
        self.language_model = language_model
        self.tfidf_output_path = tfidf_output_path
        self.tfidf_weighted = tfidf_weighted

        # pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model)
        self.model = AutoModel.from_pretrained(self.language_model)

        # tfidf weights for vector combinations
        self.tfidf = None


    def _tokenize_example(self, example: str) -> list[str]:
        """
        Params:
            - example: the text of a document

        Tokenizes the given text, creates a one-to=one mapping between token indices and tokens

        Returns:
            - A dictionary of dictionaries where "token_indices" is mapped to a dictionary containing
                the key "input_ids" mapped to token indices
        """
        example_len = len(example)
        tokens = []
        window_size = 500
        for i in range(0, example_len, window_size):
            end = i + window_size if i + window_size < example_len else example_len
            cur_tokens = self.tokenizer(
                example[i:end],
                truncation=False,
                return_attention_mask=False,
            )
            tokens.extend(cur_tokens["input_ids"])
        return tokens


    def fit(self, X, y=None):
        """
        Params:
            - X: matrix of shape [D, 1] where D is size of document set
            It is just the ['text'] column of the HuggingFace Dataset

        Tokenizes the dataset, creating tokens and token_ids
        Creates a tfidf object which can be loaded in later

        Returns:
            - self: an instance of itself to be saved for later use
        """
        token_ids = [self._tokenize_example(example) for example in X]
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in token_ids]

        # TODO: don't need to have a self.token_ids necessarily

        # print(self.token_ids[0][:10])
        # print(self.tokens[0][:10])

        self.tfidf = TFIDF(tokens)


        # print("tf:", self.tfidf.get_tf((len(self.tokens) - 1, "▁ஆக")))
        # print("idf:", self.tfidf.get_idf("▁ஆக"))
        # print("tfidf:", self.tfidf[len(self.tokens) - 1, "▁ஆக"])

        return self


    @torch.no_grad()
    def transform(self, X: list[str], y=None) -> np.ndarray:
        """
        Params:
            - X: matrix of shape [D, 1] where D is size of document set
            It is just the ['text'] column of the HuggingFace Dataset

        Returns:
            - A matrix of shape [D, E] where D is the size of the document set and E is the size of
                the document embeddings (feature size)
        """
        token_ids = [self._tokenize_example(example) for example in X]

        # TODO: retokenize the incoming X, and then use self.tfidf for tfidf weights
        word_embeddings = self.model.embeddings.word_embeddings.weight

        # vectors = list(
        #     map(
        #         lambda input_ids: self._vectorize(
        #             input_ids=input_ids,
        #             w_e=word_embeddings,
        #         ),
        #         self.token_ids
        #     )
        # )



        vectors = [
            self._vectorize(
                file_index=file_index,
                input_ids=input_ids,
                w_e=word_embeddings
            )
            for file_index, input_ids in enumerate(token_ids)
        ]

        # print("token_ids", self.token_ids[0][:10])
        # print("tokens", self.tokens[0][:10])

        print("vectors_0", vectors[0][:10])
        print("vectors_1", vectors[1][:10])

        vectors = np.array(vectors)
        print("vectors shape", vectors.shape)

        return vectors
        # return None


    # def vectorize(self):
    #     """
    #     This creates a new "vectors" entry to the given HuggingFace Dataset object. Stored
    #         in this dataset object is the binarization of a torch tensor. Each torch tensor
    #         represents the (weighted) average corresponding to the same "text" entry in the
    #         HuggingFace Dataset object.
    #     """


    @torch.no_grad()
    def _vectorize(self,
                   file_index: int,
                   input_ids: list[int],
                   w_e: torch.nn.parameter.Parameter,
                   # tfidf: TFIDF = None
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
        # print("w_e type:", type(w_e))
        tokens = [self.tokenizer.convert_ids_to_tokens(token_ids) for token_ids in input_ids]

        # grab the current tensors corresponding to the tokens in the current document
        w_i = w_e[input_ids]

        # print(w_i)

        if self.tfidf_weighted:
            # get tfidf weighted average
            document_vector = self._weighted_average(
                file_name=file_index,
                word_embeddings=w_i,
                input_ids=input_ids,
                tokens=tokens
            )
        else:
            # get unweighted average
            document_vector = torch.mean(w_i, dim=0)
            document_vector = document_vector.detach().cpu().numpy()

        print("document_vector shape", document_vector.shape)
        return document_vector


    @torch.no_grad()
    def _weighted_average(self,
                          file_name: int,
                          word_embeddings: torch.Tensor,
                          input_ids: list[int],
                          tokens: list[str]
                          ) -> torch.Tensor:
        """
        Params:
            - file_name: The file number
            - word_embeddings: A embeddings size [Document_size, word_embedding size]
            - input_ids: sequence of tokens in the given document as as integers
            - tokens: sequence of tokens in the given document
                Should have a one-to-one correspondence with input_ids

        Returns:
            - A tfidf weighted average over all tokens embeddings contained in a document
                representing a document vector
        """
        assert len(input_ids) == len(tokens)
        # print("word_embedding type:", type(word_embeddings))
        # print("word_embeddings[0] type:", type(word_embeddings[0]))
        # print("word_embeddings[0][0] type:", type(word_embeddings[0]))
        num_tokens, embedding_size = word_embeddings.size()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        document_embedding = np.zeros(embedding_size)

        for i in range(num_tokens):
            # pull current word embedding and cast to np.ndarray
            cur_embedding = word_embeddings[i].detach().cpu().numpy()

            token = tokens[i]
            token_tfidf = self.tfidf[file_name, token]
            weighted_embedding = np.multiply(token_tfidf, cur_embedding)
            document_embedding = np.add(document_embedding, weighted_embedding)

        print(f"Finished document embedding for {file_name}\n")

        return document_embedding





# if __name__ == '__main__':
#     config_file = sys.argv[1]
#     with open(config_file, "r") as ymlfile:
#         config = yaml.load(ymlfile, Loader=yaml.Loader)
#
#     ds_dict: datasets.DatasetDict = datasets.load_from_disk(config["data_path"])
#
#     document_embeddings = DocumentEmbeddings(config, ds_dict)
