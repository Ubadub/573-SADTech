import logging
from typing import Callable, Optional, Sequence

from datasets import Dataset
import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
import torch
from torch import Tensor
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils_base import BatchEncoding, VERY_LARGE_INTEGER
from transformers.utils import ModelOutput

log = logging.getLogger(__name__)


class TransformerLayerVectorizer(BaseEstimator, TransformerMixin):
    """An sklearn transformer that transforms documents (strings) into vector embeddings
    by using the CLS token position of a given set of layers of a given transformer
    language model.
    """

    def __init__(
        self,
        language_model: str = "xlm-roberta-base",
        layers_to_combine: Sequence[int] = [-1, -2, -3, -4],
        # combination_strategy: Optional[str]="concatenate",
        layer_combiner: Callable[[Sequence[ArrayLike]], NDArray] = np.hstack,
        # layer_combiner: Callable[
        #     [Sequence[ArrayLike]], NDArray
        # ] = lambda x: np.concatenate(x, axis=1),
        model_max_length: int = 512,
        device: str = "cuda:0",
        # model_max_length: Optional[int] = None,
    ):
        """Constructor.

        Args:
            lm_name_or_path:
                Identifier (name in HuggingFaceHub or path to saved model) for a
                pretrained model (and associated tokenizer) that can be loaded by
                HuggingFace's `from_pretrained()` methods.

                Let `(embedding_dim, hidden_dim)` be the shape of a hidden layer of
                this model:

                    `embedding_dim`: the maximum length accepted by the model; documents
                    with more tokens than this will be truncated, while shorter
                    documents will be padded. The size of this dimension is not directly
                    relevant to the shapes of the output embeddings, since only the
                    vector corresponding to the CLS token (typically, the 0th index) is
                    used to compute the embeddings.

                    `hidden_dim`: The second dimension of one of the hidden layers,
                    equivalent to the first/only dimension of the pooler output.

                Then, given a set of D input documents, calling `fit_transform`
                or `transform` on this class will embed each of the D documents as an
                `NDArray` of shape `(n_features,)`, where `n_features` is defined by
                `layers` and `layer_combiner`. The resulting matrix will thus be of
                shape `(D, n_features)`.

                As the default `layer_combiner` is the concatenation operation, the
                default behavior is for n_features to be equal to `n_layers *
                hidden_dim.` However, other operations are possible: for example,
                averaging the layers would yield a per-document embedding of shape
                `(hidden_dim,)`.


                Defines the following parameters which define the shape of the vectors
                that are inputs and outputs of this transformer:

                Thus, given a set of D input documents, this transformer will embed each
                of the D documents as an `NDArray` of shape (
                Thus, for a given set of documents whose longest document = the maximum
                model input size = embedding_dim, Contains hidden layers of shape
            layers:
                A `Sequence` (e.g. an array or tuple) indicating which of the hidden
                layers to use in vectorization, of length `n_layers`.

                Can consist of any integer identifiers that could be used to index an
                array-like instance. For example, can consist of negative indices to
                indicate the last X layers of a model.
            layer_combiner:
                A function (`Callable`) that takes in an `NDArray` of shape
                `(n_layers, D, hidden_dim)` and returns and combines them into a
                single `NDArray` of shape `(n_features,)`.

                This function is used to determine how to combine the `n_layers`
                vectors of the given `layers` of the language model into a
                single-dimensional vector. The default and prototypical function is
                `np.concatenate` (along the first axis, corresponding to the layers).
            model_max_length:
                ...
        """
        super().__init__()
        self.language_model: str = language_model
        self.device: str = device

        # if model_max_length is None:
        #     model_max_length = getattr(
        #         self.lm_.config, "max_position_embeddings", VERY_LARGE_INTEGER
        #     )

        self.model_max_length = model_max_length
        # self.model_max_length = self.tokenizer_.max_len_single_sentence
        # tokenizer_max_model_length = self.tokenizer_.max_model_input_sizes.get(lm_name_or_path)
        # self._model_max_length = self.lm_.config.max_position_embeddings
        # max_position_embeddings = getattr(self.lm_.config, "max_position_embeddings", None)

        # if model_max_length is None:
        #                if self.self.tokenizer_.max_model_input_sizes

        #    self._model_max_length = self.getattr(self.lm_.config, "max_position_embeddings", 0)

        self.layers_to_combine = layers_to_combine
        # self.combination_strategy = combination_strategy
        self.layer_combiner = layer_combiner

    def _tokenize_docs(self, docs) -> BatchEncoding:
        return self.tokenizer_(
            list(docs),
            is_split_into_words=False,
            max_length=self.model_max_length,
            padding="max_length",
            # padding=True,
            return_tensors="pt",
            truncation=True,
        )

    def _model_outputs(self, docs) -> ModelOutput:
        model_inputs: BatchEncoding = self._tokenize_docs(docs).to(self.device)
        log.info(f"model_inputs.input_ids.shape: {model_inputs.input_ids.shape}")
        return self.lm_(**model_inputs, output_hidden_states=True)

    # model_outputs: Iterable[ModelOutput] = (
    #     self._lm(i, **self._lm_kwargs) for i in model_inputs
    # )

    def fit(self, X, y=None):
        self.tokenizer_: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            self.language_model,
            model_max_length=self.model_max_length,
        )
        self.lm_: PreTrainedModel = AutoModel.from_pretrained(self.language_model)
        self.lm_kwargs_: dict = {"output_hidden_states": True}
        return self

    @torch.no_grad()
    def transform(self, X, y=None) -> NDArray:
        """Vectorize the given dataset using the CLS token vectors of the given layers.

        Given a dataset of N entries, returns an `NDArray` instance of shape (N,M)
        where M is one of the dimensions of a hidden layer of the transformer language
        model corresponding to `self._model` (the `strategy` arguument of the class
        constructor determines the exact algorithm used).

        Args:
            X: array-like of shape (n_samples, n_features)
                Input samples.

            y:  Not used and not required, because this is an unsupervised
            transformation. default=None

        Returns:
            X_new: ndarray array of shape (n_samples, n_features)
                `n_features` is determined by `hidden_dim` and `self.layer_combiner`
                (see class docs for more info).
        """
        self.lm_.to(self.device)
        model_outputs: ModelOutput = self._model_outputs(X)
        n_layers = len(self.layers_to_combine)
        layers = (
            model_outputs.hidden_states[i].detach().cpu().numpy()
            for i in self.layers_to_combine
        )
        # print("Length of layers:", len(layers), "Shape of elements:", layers[0].shape)
        cls_layers_by_doc_by_feats = np.stack([layer[:, 0] for layer in layers])
        vecs = self.layer_combiner(cls_layers_by_doc_by_feats)
        # print("Return vectors shape:", vecs.shape)
        self.lm_.to("cpu")
        return vecs
