from functools import wraps
import logging
import os
import pickle
import re
from typing import Any, Callable, ParamSpec, TypeVar

import hydra
from imblearn.pipeline import Pipeline
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

T = TypeVar("T")
P = ParamSpec("P")

ITEM_SEP = r"%"
KV_SEP = r"="
CFG_CHANGE_SYMS = "+~"
CFG_PKG_GRP_SEP = "@"
CFG_PKG_SEPS = "./"

CFG_TRANSFORMERS = "transformers"
CFG_PRERESAMPLE_TRANSFORMERS = "preresample_transformers"
CFG_RESAMPLERS = "resamplers"
CFG_POSTRESAMPLE_TRANSFORMERS = "postresample_transformers"

log = logging.getLogger(__name__)


def exception_handler(
    exception_action: Callable[[Exception], Any]
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(wrapped_func: Callable[P, T]) -> Callable[P, T]:
        @wraps(wrapped_func)
        def _(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return wrapped_func(*args, **kwargs)
            except Exception as e:
                exception_action(e)
                raise e

        return _

    return decorator


slash2under = lambda x: x.replace(os.sep, "_")


def clean_path(path: str) -> str:
    args = (_.lstrip(CFG_CHANGE_SYMS) for _ in path.split(ITEM_SEP))
    args_cleaned = (_.split(CFG_PKG_GRP_SEP)[-1] for _ in args)
    sorted_args = sorted(args_cleaned, reverse=True)
    arg_pieces = (
        re.findall(
            rf"[{CFG_PKG_SEPS}]?([^{CFG_PKG_SEPS}]*){KV_SEP}(.+)$|^([^{KV_SEP}]+)$", _
        )
        for _ in sorted_args
    )
    flattened = (match_grp for __ in arg_pieces for match_grp in __)
    no_empties_no_slashes = ((slash2under(_) for _ in __ if _) for __ in flattened)
    return os.sep.join(KV_SEP.join(_) for _ in no_empties_no_slashes)
    # arg_pieces_no_empties = ((
    # matches = re.findall(f"{KV_SEP}(.*?)(?:{ITEM_SEP}|$)", path)
    # if matches:
    #     return os.sep.join([slash2under(_) for _ in matches])
    # else:
    #     return slash2under(path)


OmegaConf.register_new_resolver("clean_path", clean_path)
OmegaConf.register_new_resolver("slash2under", slash2under)
# OmegaConf.register_new_resolver("clean_path", lambda x: x.split("=")[-1] if "=" in x else x)

OmegaConf.register_new_resolver("ref", lambda x: f"${{ref:{x}}}")


def resolve_conf_crossrefs(conf):
    """Resolve references between objects, such that they are not instatiated multiple times.
    Searches in the root first and then in the parent"""
    conf = OmegaConf.to_container(conf)
    resolved_dict = _resolve_obj_ref(conf, conf, conf)
    return DictConfig(resolved_dict, flags={"allow_objects": True})


def _resolve_obj_ref(conf, root, parent):
    if isinstance(conf, dict):
        return {k: _resolve_obj_ref(v, root, conf) for k, v in conf.items()}
    elif isinstance(conf, list):
        return [_resolve_obj_ref(v, root, conf) for v in conf]
    elif isinstance(conf, str) and conf.startswith("${ref:"):
        key = conf[6:-1]
        return __resolve_obj_ref(key, root, parent)
    else:
        return conf


def __resolve_obj_ref(v: str, root: dict, parent: dict) -> Any:
    parts = v.split(".")
    for conf in (root, parent):
        for part in parts:
            if part in conf:
                conf = conf[part]
            else:
                break
        else:
            return conf
    return v


def setup_rng(cfg: DictConfig) -> DictConfig:
    cfg._set_flag("allow_objects", True)
    # cfg = DictConfig(cfg, flags={"allow_objects": True})
    cfg.global_rng = hydra.utils.instantiate(cfg.global_rng)
    return resolve_conf_crossrefs(cfg)


def load_pipeline(path: str) -> Pipeline:
    # in_path = os.path.join(clf_or_saved_path, f"fold{n}.model")
    log.info(f"Loading model from: {path}")
    with open(path, "rb") as f:
        clf = pickle.load(f)

    assert isinstance(
        clf, Pipeline
    ), f"Tried to unpickle Pipeline object, but got object of type: {type(clf)}"

    return clf


def assemble_pipeline(
    pipeline_cfg: DictConfig,
) -> Pipeline:
    column_transformers = []

    if "text" in pipeline_cfg:
        text_cfg = pipeline_cfg.text
        text_col = text_cfg.column_name
        # audio_preprocessors = pipeline_cfg.audio

        # text_transformers_cfg = pipeline_cfg.get(CFG_TEXT_TRANSFORMERS, [])
        text_transformer = Pipeline(
            steps=[
                ("vectorizer", text_cfg.vectorizer),  # mandatory
                *(
                    (tr_name, tr)
                    for tr_name, tr in text_cfg.get(CFG_TRANSFORMERS, {}).items()
                ),
            ],
            # memory="sklearn_cache/text_transformer",
        )
        log.debug(f"Text transformer: {text_transformer}")

        column_transformers.append((text_col, text_transformer, text_col))

    if "audio" in pipeline_cfg:
        audio_cfg = pipeline_cfg.audio
        audio_col = audio_cfg.column_name
        log.debug(f"Audio column: {audio_col}")
        column_transformers.append(
            ("audio", FunctionTransformer(func=np.vstack), audio_col)
        )

        # audio_transformer = Pipeline(
        #     steps=[
        #         audio_preprocessors.vectorizer,  # mandatory
        #         *(
        #             (tr_name, tr)
        #             for tr_name, tr in audio_preprocessors.get(CFG_TRANSFORMERS, {}).items()
        #         ),
        #     ],
        # )

        # log.debug(f"Audio transformer: {audio_transformer}")

    if "text" not in pipeline_cfg and "audio" not in pipeline_cfg:
        raise ValueError(
            "Must specify at least one of text or audio in pipeline config."
        )

    preprocessor = ColumnTransformer(
        transformers=column_transformers,
        n_jobs=-1,
    )

    log.debug(f"Preprocessor: {preprocessor}")

    preresample_transformers = [
        (tr_name, tr)
        for tr_name, tr in pipeline_cfg.get(CFG_PRERESAMPLE_TRANSFORMERS, {}).items()
    ]

    log.debug(f"Preresample transformers: {preresample_transformers}")

    resamplers = [
        (resampler_name, resampler)
        for resampler_name, resampler in pipeline_cfg.get(CFG_RESAMPLERS, {}).items()
    ]

    log.debug(f"Resamplers: {resamplers}")

    postresample_transformers = [
        (tr_name, tr)
        for tr_name, tr in pipeline_cfg.get(CFG_POSTRESAMPLE_TRANSFORMERS, {}).items()
    ]

    log.debug(f"Postresample transformers: {postresample_transformers}")

    classifier = pipeline_cfg.classifier  # mandatory

    log.debug(f"Classifier: {classifier}")

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            *preresample_transformers,
            *resamplers,
            *postresample_transformers,
            ("classifier", classifier),
        ],
        # memory=classifier_cache,
    )

    log.info(f"Pipeline: {clf}")

    return clf
