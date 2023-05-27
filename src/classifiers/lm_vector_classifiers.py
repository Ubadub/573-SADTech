from typing import Iterable, Optional

# from sklearn.linear_model import LogisticRegression
from datasets import Dataset
import numpy as np
import torch
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.utils import ModelOutput

from .classifier import Classifier


class TransformerLayerVectorClassifier(Classifier):
    """ """

    TOKENIZER_ARGS_FIELD = "TokenizerArguments"
    TOKENIZER_KWARGS_FIELD = "TokenizerKeywordArguments"

    def __init__(
        self,
        config: dict,
        lm_name_or_path: str,
        strategy: Optional[str],
        # lm_kwargs: dict = {},
    ):
        """
        Constructor.

        Args:
            config:
                A dictionary (e.g. created from a YAML config file) containing various
                configuration settings and variables.
                See superclass for generally required and optional contents of config
                file.
                Optionally may also contain (key : value):
                    `TransformerLayerVectorClassifier.TOKENIZER_ARGS_FIELD`:
                        any arguments to pass to the tokenizer during tokenization.
                    `TransformerLayerVectorClassifier.TOKENIZER_KWARGS_FIELD`:
                        any keyword arguments to pass to the tokenizer during
                        tokenization.
            lm_name_or_path:
                Identifier (name in HuggingFaceHub or path to saved model) for a
                pretrained model (and associated tokenized) that can be loaded by
                HuggingFace's `from_pretrained()` methods.
            strategy:
                Defines the vectorization strategy to do. Choose from:
                    TODO
        """
        super().__init__(config=config)
        self._strat = strategy
        self._transformer_lm: PreTrainedModel = AutoModel.from_pretrained(
            lm_name_or_path
        )
        self._lm_kwargs: dict = {"output_hidden_states": bool(self._strat)}
        self._tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            lm_name_or_path
        )
        self._tokenizer_args: Iterable = (
            self._cfg.get(self.TOKENIZER_ARGS_FIELD, []) or []
        )
        self._tokenizer_kwargs: dict = (
            self._cfg.get(self.TOKENIZER_KWARGS_FIELD, {}) or {}
        )
        # self._tokenizer_kwargs  = {
        #     return_tensors="pt",
        #     max_length=512,
        #     truncation=True
        # }
        # self._model: BaseEstimator = self._build_model(*model_args, **model_kwargs)

    @torch.no_grad()
    # def _vectorize(self, ds: Dataset, text_field: str = "text", model_inputs_field: str = "input_ids", **tokenizer_kwargs):
    def _vectorize(
        self,
        ds: Dataset,
        text_field: str = "text",
        model_inputs_field: str = "input_ids",
    ) -> np.ndarray:
        """Vectorize the given dataset using the CLS token vectors in the last N layers
        of the given model using tthe class-defined strategy.

        Given a dataset of N entries, returns an `np.ndarray` instance of shape (N,M)
        where M is one of the dimensions of a hidden layer of the transformer language
        model corresponding to `self._model` (the `strategy` arguument of the class
        constructor determines the exact algorithm used).

        Args:
            ds:
                The dataset to use, containing num_instances rows.
            text_field:
                name of the field in ds that holds the text to vectorize.
            model_inputs_field:
                name of the field in tokenizer output that holds the inputs to the model

        Returns:
            An `np.ndarray` instance of shape (num_instances, num_features) where
            num_features is determined by the vectorization algorithm.
        """
        # with torch.no_grad():
        # model_inputs: Iterable[Tensor] = (self._tokenizer(text, return_tensors="pt", **tokenizer_kwargs)[model_inputs_field] for text in ds[text_field])
        model_inputs: Iterable[Tensor] = (
            self._tokenizer(text, *self._tokenizer_args, **self._tokenizer_kwargs)[
                model_inputs_field
            ]
            for text in ds[text_field]
        )
        model_outputs: Iterable[ModelOutput] = (
            self._transformer_lm(i, **self._lm_kwargs) for i in model_inputs
        )
        last_n_layers = (
            (tensor.detach().cpu().squeeze().numpy() for tensor in o.hidden_states[-4:])
            for o in model_outputs
        )
        last_n_layers_cls_token = ((_[0] for _ in l) for l in last_n_layers)
        vecs = tuple(np.concatenate(tuple(_)) for _ in last_n_layers_cls_token)
        return vecs

    # @abstractmethod
    # def _build_model(self, *args, **kwargs) -> BaseEstimator:
    #     """ """


# class TLVLogisticRegressionClassifier(TransformerLayerVectorClassifier):
#     def _build_model(self, *args, **kwargs) -> BaseEstimator:
#         """ """
#         return LogisticRegression(
#             *args, class_weight="balanced", max_iter=1000, **kwargs
#         )
