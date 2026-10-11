"""設定(YAML 1ファイル)の読み込みと上書き。"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_CONFIG = _CONFIG_DIR / "solo_ppo.yaml"
# 設定の mode ごとの既定値ファイル(mode が無ければ solo)
DEFAULT_CONFIGS = {"solo": DEFAULT_CONFIG, "versus": _CONFIG_DIR / "versus_ppo.yaml"}


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    """YAML を読み、``a.b.c=値`` 形式の上書きを適用する(値は YAML として解釈)。"""
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    default = DEFAULT_CONFIGS[cfg.get("mode", "solo")]
    if path.resolve() != default.resolve() and default.exists():
        # 古いランの config.yaml に無い(後から増えた)キーは既定値で補う
        with open(default, encoding="utf-8") as f:
            cfg = _merge_defaults(yaml.safe_load(f), cfg)
    for ov in overrides or []:
        key, sep, raw = ov.partition("=")
        if not sep:
            raise ValueError(f"override must be key=value: {ov!r}")
        set_by_path(cfg, key.strip(), yaml.safe_load(raw))
    return cfg


def _merge_defaults(default: dict, cfg: dict) -> dict:
    out = copy.deepcopy(cfg)
    for k, v in default.items():
        if k not in out:
            out[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(out[k], dict):
            out[k] = _merge_defaults(v, out[k])
    return out


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    node = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise KeyError(f"unknown config section: {dotted}")
        node = node[k]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key: {dotted}")
    node[keys[-1]] = value


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def eval_seeds(cfg: dict) -> list[int]:
    s = cfg["eval"]["seeds"]
    return list(range(int(s["start"]), int(s["start"]) + int(s["count"])))


def deep_copy(cfg: dict) -> dict:
    return copy.deepcopy(cfg)
