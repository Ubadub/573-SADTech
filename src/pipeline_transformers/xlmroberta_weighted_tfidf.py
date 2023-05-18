"""
Document Embeddings: (Optional) TFIDF Weighted Average Word Embeddings

Example run:
python3 -m pipeline_transformers -d ../data/tam/train_dataset_dict train
    -c config/pipeline/xlmroberta_weighted_tfidf.yml -m ../outputs/debug/tam
"""

import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator, TransformerMixin


class TFIDF:
    """
    Creates a TFIDF object, storing term frequencies, and inverse document frequencies
    """

    TFIDF_UNK = "<@tfidf_unk$>"

    def __init__(
        self,
        data: list[list[str]],
    ) -> None:
        """
            Params:
                - data: A data set represented by a list of list of tokens. Each inner list represents
                    one document in the data set.

        Initializes a TFIDF object
        """
        self.data = data
        self.N = len(self.data)

        self.idf = {self.TFIDF_UNK: 0}
        self.tf = {}

        # calculate tfidf scores
        self._obtain_tf_and_idf_counts()
        self._obtain_idf()

    def _obtain_tf_and_idf_counts(self) -> None:
        """
        Creates term-frequency (tf) counts
        Creates document frequency counts
        """
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

    def _obtain_idf(self) -> None:
        """
        Transforms document frequency counts into inverse document frequency (idf)
        """
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


class DocumentEmbeddings(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        language_model: str,
        tfidf_weighted: bool = True,
    ) -> None:
        """
        Params:
            - language_model: the pretrained language model (e.g. XLM-Roberta)
            - tfidf_weighted: whether or not the resulting document vectors are combined using
                tfidf weights

        Initializes a DocumentEmbeddings object for use in Sklearn's pipeline. This object creates
            document vectors as (tfidf weighted) averaged word embeddings for use in downstream tasks
        """
        super().__init__()
        self.language_model = language_model
        self.tfidf_weighted = tfidf_weighted

        # pretrained model
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model)
        self.model = AutoModel.from_pretrained(self.language_model)

    def _tokenize_example(self, example: str) -> list[int]:
        """
        Params:
            - example: the text of a document

        Tokenizes the given text, creates a one-to=one mapping between token indices and tokens

        Returns:
            - A list of tokens, where each token is represented by its corresponding index
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
        Creates a tfidf object which can be loaded in for later use in
            Sklearn's transform() function

        Returns:
            - self: an instance of itself to be saved for later use
        """
        token_ids = [self._tokenize_example(example) for example in X]
        tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in token_ids]

        self.tfidf_ = TFIDF(tokens)

        return self

    @torch.no_grad()
    def transform(self, X: list[str], y=None) -> np.ndarray:
        """
        Params:
            - X: matrix of shape [D, 1] where D is size of document set
            It is just like the ['text'] column of the HuggingFace Dataset

        This creates document embeddings using the (tfidf weighted) average of the
            word vectors from the pretrained language model

        Returns:
            - A matrix of shape [D, E] where D is the size of the document set and E is the size of
                the document embeddings (feature size)
        """
        token_ids = [self._tokenize_example(example) for example in X]

        word_embeddings = self.model.embeddings.word_embeddings.weight

        vectors = [
            self._vectorize(
                file_index=file_index, input_ids=input_ids, w_e=word_embeddings
            )
            for file_index, input_ids in enumerate(token_ids)
        ]

        return np.array(vectors)

    @torch.no_grad()
    def _vectorize(
        self,
        file_index: int,
        input_ids: list[int],
        w_e: torch.nn.parameter.Parameter,
    ) -> None:
        """
        Params:
            - file_index: the index of the current document
            - input_ids: A list of tokens as indices representing the current document
            - w_e: The token embeddings obtained from the pretrained model's word embedding layer

        Creates a (tfidf weighted) average of the token vectors constituting the
            given entry of the HuggingFace dataset.
        """
        # grab the current tensors corresponding to the tokens in the current document
        w_i = w_e[input_ids]

        if self.tfidf_weighted:
            # get tfidf weighted average
            document_vector = self._weighted_average(
                file_name=file_index,
                word_embeddings=w_i,
                input_ids=input_ids,
            )
        else:
            # get unweighted average
            document_vector = torch.mean(w_i, dim=0)
            document_vector = document_vector.detach().cpu().numpy()

        return document_vector

    @torch.no_grad()
    def _weighted_average(
        self,
        file_name: int,
        word_embeddings: torch.Tensor,
        input_ids: list[int],
    ) -> torch.Tensor:
        """
        Params:
            - file_name: The file number
            - word_embeddings: A embedding size [Document_size, word_embedding size]
            - input_ids: sequence of tokens in the given document as as integers
            - tokens: sequence of tokens in the given document
                Should have a one-to-one correspondence with input_ids

        Returns:
            - A tfidf weighted average over all tokens embeddings contained in a document
                representing a document vector
        """
        num_tokens, embedding_size = word_embeddings.size()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        document_embedding = np.zeros(embedding_size)

        for i in range(num_tokens):
            # pull current word embedding and cast to np.ndarray
            cur_embedding = word_embeddings[i].detach().cpu().numpy()

            token = tokens[i]
            token_tfidf = self.tfidf_[file_name, token]
            weighted_embedding = np.multiply(token_tfidf, cur_embedding)
            document_embedding = np.add(document_embedding, weighted_embedding)

        return document_embedding
