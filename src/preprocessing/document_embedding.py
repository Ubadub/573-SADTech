"""
Document Embeddings: (TFIDF Weighted) Average Word Embeddings
"""

import sys
import yaml

import datasets
import torch
import pickle
from transformers import AutoTokenizer, AutoModel


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
        self.data = ds_dict["train"] if "train" in ds_dict else ds_dict["test"]
        self.pretrained_model = config["pretrained_model"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.tokenize_dataset()

        self.model = AutoModel.from_pretrained(self.pretrained_model)
        self.vectorize()

        print(self.data)


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


    @torch.no_grad()
    def vectorize(self):
        """
        This creates a new "vectors" entry to the given HuggingFace Dataset object. Stored
            in this dataset object is the binarization of a torch tensor. Each torch tensor
            represents the (weighted) average corresponding to the same "text" entry in the
            HuggingFace Dataset object.
        """
        word_embeddings = self.model.embeddings.word_embeddings.weight

        self.data = self.data.map(
            self._vectorize_helper,
            fn_kwargs={
                "w_e": word_embeddings,
            }
        )


    @torch.no_grad()
    def _vectorize_helper(self,
                          example: datasets.formatting.formatting.LazyRow,
                          w_e: torch.nn.parameter.Parameter
                          ) -> None:
        """
            Params:
                - example: An entry of the HuggingFace Dataset object
                - w_e: The word embeddings obtained from the pretrained model

            Takes an entry of a HuggingFace dataset, as well as the word embedding layer of the
                pretrained model. Creates a (weighted) average of the token vectors constituting the
                given entry of the HuggingFace dataset.
        """
        # grab the current indices mapped to the tokens that make up the current document
        input_ids = example["token_indices"]["input_ids"]

        # grab the current tensors corresponding to the tokens in the current document
        w_i = w_e[input_ids]

        if self.config["tfidf_weighted"]:
            # get tfidf weighted average
            vectors = self._weighted_average(w_i)
        else:
            # get unweighted average
            vectors = torch.mean(w_i, dim=0)

        # transform torch tensor into numpy array
        vectors = vectors.detach().cpu().numpy()

        vectors = pickle.dumps(vectors, protocol=pickle.HIGHEST_PROTOCOL)

        return {"vectors": vectors}


    def _weighted_average(self, w_i: torch.nn.parameter.Parameter) -> torch.nn.parameter.Parameter:
        """
        TODO: Create TFIDF weights, and then weight the tensors w_i by the tf-idf weights
        """
        raise NotImplementedError




if __name__ == '__main__':
    config_file = sys.argv[1]
    with open(config_file, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.Loader)

    ds_dict: datasets.DatasetDict = datasets.load_from_disk(config["data_path"])

    document_embeddings = DocumentEmbeddings(config, ds_dict)
