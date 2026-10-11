"""観測空間の定義。

観測は dict:

- ``board``: ``(2, 6, 13, 6)`` uint8。[自分, 相手] × 平面[空, 赤, 緑, 青, 黄, おじゃま]
  × 段(0 = 1段目, 12 = 13段目) × 列(0 = 左端)。
  1人用では相手側はすべて0。
- ``pieces``: ``(3, 2, 4)`` uint8。[現在, ネクスト, ネクネク] × [軸, 子] × 色のワンホット。
- ``ojama``: ``(2,)`` int32。[自分, 相手] の予告おじゃま数。
- ``action_mask``: ``(22,)`` int8。合法手なら1。
"""

import numpy as np
from gymnasium import spaces

from ._puyo import (
    HEIGHT,
    NUM_ACTION_FEATURES,
    NUM_ACTIONS,
    NUM_COLORS,
    NUM_VERSUS_ACTION_FEATURES,
    NUM_VERSUS_INFO,
    WIDTH,
)

NUM_BOARD_PLANES = NUM_COLORS + 2
NUM_VISIBLE_PAIRS = 3


def observation_space(action_features: bool = False, versus: bool = False) -> spaces.Dict:
    """``action_features`` が True なら、手ごとの結果の特徴量
    ``action_features``: ``(22, NUM_ACTION_FEATURES)`` float32(``Game.action_features``)を加える。

    ``versus`` が True なら、手ごとの特徴量は対戦用の ``(22, NUM_VERSUS_ACTION_FEATURES)``
    (``Versus.versus_action_features``)になり、対局全体の情報
    ``versus_info``: ``(NUM_VERSUS_INFO,)`` float32(``Versus.versus_info``)も加える。
    """
    d = {
        "board": spaces.Box(0, 1, (2, NUM_BOARD_PLANES, HEIGHT, WIDTH), dtype=np.uint8),
        "pieces": spaces.Box(0, 1, (NUM_VISIBLE_PAIRS, 2, NUM_COLORS), dtype=np.uint8),
        "ojama": spaces.Box(0, np.iinfo(np.int32).max, (2,), dtype=np.int32),
        "action_mask": spaces.Box(0, 1, (NUM_ACTIONS,), dtype=np.int8),
    }
    if versus:
        d["action_features"] = spaces.Box(
            0.0, 10.0, (NUM_ACTIONS, NUM_VERSUS_ACTION_FEATURES), dtype=np.float32
        )
        d["versus_info"] = spaces.Box(0.0, 10.0, (NUM_VERSUS_INFO,), dtype=np.float32)
    elif action_features:
        d["action_features"] = spaces.Box(
            0.0, 10.0, (NUM_ACTIONS, NUM_ACTION_FEATURES), dtype=np.float32
        )
    return spaces.Dict(d)


def action_space() -> spaces.Discrete:
    return spaces.Discrete(NUM_ACTIONS)
