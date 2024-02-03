import os
import re
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

ITEM_SEP = r","
KV_SEP = r"="

slash2under = lambda x: x.replace(os.sep, "_")

def clean_path(path: str) -> str:
    matches = re.findall(f"{KV_SEP}(.*?)(?:{ITEM_SEP}|$)", path)
    if matches:
        return os.sep.join([slash2under(_) for _ in matches])
    else:
        return slash2under(path)

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

