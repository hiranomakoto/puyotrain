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

from ._puyo import HEIGHT, NUM_ACTIONS, NUM_COLORS, WIDTH

NUM_BOARD_PLANES = NUM_COLORS + 2
NUM_VISIBLE_PAIRS = 3


def observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "board": spaces.Box(0, 1, (2, NUM_BOARD_PLANES, HEIGHT, WIDTH), dtype=np.uint8),
            "pieces": spaces.Box(0, 1, (NUM_VISIBLE_PAIRS, 2, NUM_COLORS), dtype=np.uint8),
            "ojama": spaces.Box(0, np.iinfo(np.int32).max, (2,), dtype=np.int32),
            "action_mask": spaces.Box(0, 1, (NUM_ACTIONS,), dtype=np.int8),
        }
    )


def action_space() -> spaces.Discrete:
    return spaces.Discrete(NUM_ACTIONS)
